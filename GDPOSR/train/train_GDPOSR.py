import os
import gc
import lpips
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import copy

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats
import sys
sys.path.append("GDPOSR")
from modelfile.GDPOSR import GDPOSR as GDPOSRModel
from my_utils.training_utils_realsr import parse_args_realsr_training, PairedSROnlineDataset  

from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

sys.path.append('GDPOSR')
from GDPOSR.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from diffusers.training_utils import compute_snr
from diffusers import DDPMScheduler, AutoencoderKL
from GDPOSR.losses.grpo import AdaptiveReward as RewardFunction

from ram.models.ram_lora import ram
from ram import inference_ram as inference


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    net_pix2pix = GDPOSRModel(args)
    net_pix2pix.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_pix2pix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_pix2pix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)
    net_ARF = RewardFunction()
    net_ARF.requires_grad_(False)

    # # set adapter
    net_pix2pix.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])

    # make the optimizer
    layers_to_opt = []
    for n, _p in net_pix2pix.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    # make the dataloader
    dataset_train = PairedSROnlineDataset(dataset_folder=args.dataset_folder, image_prep=args.train_image_prep, split="train", deg_file_path=args.deg_file_path, args=args)
    dataset_val = PairedSROnlineDataset(dataset_folder=args.dataset_folder, image_prep=args.test_image_prep, split="test", deg_file_path=args.deg_file_path, args=args)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # init RAM
    ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    RAM = ram(pretrained='./ckp/ram_swin_large_14m.pth',
            pretrained_condition=None,
            image_size=384,
            vit='swin_l')
    RAM.eval()
    RAM.to("cuda", dtype=torch.float16)

    # Prepare everything with our `accelerator`.
    net_pix2pix, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_pix2pix, optimizer, dl_train, lr_scheduler
    )
    net_lpips, net_ARF = accelerator.prepare(net_lpips, net_ARF)
    # renorm with image net statistics
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # start the training loop
    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            with accelerator.accumulate(net_pix2pix):
                x_src = batch["LR"]
                x_tgt = batch["HR"]
                fedilty_ratio = batch["fedilty_ratio"]
                detail_ratio = batch["detail_ratio"]

                B, C, H, W = x_src.shape
                # image description
                x_tgt_ram = ram_transforms(x_tgt*0.5+0.5)
                caption_r = inference(x_tgt_ram.to(dtype=torch.float16), RAM)
                with torch.no_grad():
                    positive_prompt = []
                    negative_prompt = []
                    for i in range(B):
                        ram_image = x_tgt[i,:,:,:].unsqueeze(0)
                        x_tgt_ram = ram_transforms(ram_image*0.5+0.5)
                        caption = inference(x_tgt_ram.to(dtype=torch.float16), RAM)
                        positive_prompt.append(f'{caption[0]}, {args.positive_prompt}')
                        negative_prompt.append(args.negative_prompt)
                # generate some samples
                if torch.cuda.device_count() > 1:
                    sample_images, _, _ = net_pix2pix.module.GDPOReference(x_src, positive_prompt=positive_prompt, negative_prompt=negative_prompt, args=args, groupsize=args.groupsize)
                else:
                    sample_images, _, _ = net_pix2pix.GDPOReference(x_src, positive_prompt=positive_prompt, negative_prompt=negative_prompt, args=args, groupsize=args.groupsize)
                # select winning and losing samples:
                x_tgt_re = x_tgt.unsqueeze(1).repeat(1,args.groupsize,1,1,1)
                rewards = net_ARF(sample_images, x_tgt_re, fedilty_ratio, detail_ratio)
                rewards = rewards.cuda()
                b_sample, g_sample, c_sample, h_sample, w_sample = sample_images.shape
                x_src_wl = sample_images.view(b_sample*g_sample, c_sample, h_sample, w_sample)
                ps_wl = []
                nps_wl = []
                for i in range(args.groupsize):
                    ps_wl += positive_prompt
                    nps_wl += negative_prompt
                # forward pass
                x_tgt_pred, latents_pred, model_pred, prompt_embeds, neg_prompt_embeds, noise, ref_output_image, ref_x_denoised, ref_model_pred = net_pix2pix(x_src_wl, positive_prompt=ps_wl, negative_prompt=nps_wl, args=args)
                # GDPO
                model_losses = (model_pred - noise).pow(2).mean(dim=[1,2,3])
                # b_model, c_model, h_model, w_model = model_losses.shape
                model_losses = model_losses.view(b_sample, g_sample)
                model_losses = rewards * model_losses
                model_diff = model_losses.sum(1)
                # model_losses_w, model_losses_l = model_losses.chunk(2)
                ref_losses = (ref_model_pred - noise).pow(2).mean(dim=[1,2,3])
                ref_losses = ref_losses.view(b_sample, g_sample)
                ref_losses = rewards * ref_losses
                ref_diff = ref_losses.sum(1)
                scale_term = -0.5 * 5000 
                inside_term = scale_term * (model_diff - ref_diff)
                implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                gdpo_loss = -1 * F.logsigmoid(inside_term).mean()
                loss = gdpo_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["loss"] = gdpo_loss.detach().item()
                    progress_bar.set_postfix(**logs)

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_pix2pix).save_model(outf)

                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_realsr_training()
    main(args)
