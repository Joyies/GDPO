import os
import requests
import sys
import copy
import random
import time
import glob
import math
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from peft import LoraConfig
from types import SimpleNamespace
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.utils.import_utils import is_xformers_available

def make_1step_sched(pretrained_model_path):
    noise_scheduler_1step = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step

def find_filepath(directory, filename):
    matches = glob.glob(f"{directory}/**/{filename}", recursive=True)
    return matches[0] if matches else None


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def initialize_vae(rank, return_lora_module_names=False, pretrained_model_name_or_path=None):
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()
    
    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["conv1","conv2","conv_in", "conv_shortcut",
        "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0",
    ]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n: continue
        for pattern in l_grep:
            if pattern in n and ("encoder" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and ("decoder" in n):
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif 'post_quant_conv' in n:
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break
    lora_conf_encoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder)
    vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    vae.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    # vae.set_adapter(["default_encoder", "default_decoder"])
    if return_lora_module_names:
        return vae, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
    else:
        return vae

def initialize_unet(rank, return_lora_module_names=False, pretrained_model_name_or_path=None):
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n: continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and "up_blocks" in n:
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break
    lora_conf_encoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")
    if return_lora_module_names:
        return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
    else:
        return unet

def initialize_unet_sr(rank, return_lora_module_names=False, pretrained_model_name_or_path=None, args=None):
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    if args.use_lr_concat_lr_999noise:
        new_conv_in = torch.nn.Conv2d(8, 320, 3, 1, 1)
        new_conv_in.weight.data[:, :4, ...] = unet.conv_in.weight.data
        new_conv_in.weight.data[:, -4:, ...] = unet.conv_in.weight.data
        new_conv_in.bias.data = unet.conv_in.bias.data
        unet.conv_in = new_conv_in
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n: continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and "up_blocks" in n:
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break
    lora_conf_encoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")
    if return_lora_module_names:
        return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
    else:
        return unet

class VSD(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__() 

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.sched = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        self.unet_fix = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
        self.unet_update, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others =\
                initialize_unet(rank=args.lora_rank_unet_vsd, pretrained_model_name_or_path=args.pretrained_model_name_or_path, return_lora_module_names=True)
        self.lora_rank_unet = args.lora_rank_unet_vsd

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet_fix.enable_xformers_memory_efficient_attention()
                self.unet_update.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available, please install it by running `pip install xformers`")

        if args.gradient_checkpointing:
            self.unet_fix.enable_gradient_checkpointing()
            self.unet_update.enable_gradient_checkpointing()

        self.text_encoder.to(accelerator.device, dtype=weight_dtype)
        self.unet_fix.to(accelerator.device, dtype=weight_dtype)
        self.unet_update.to(accelerator.device)
        self.vae.to(accelerator.device)
        
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet_fix.requires_grad_(False)

    def set_eval(self):
        self.unet_fix.eval()
        self.unet.eval()
        self.unet_update.eval()

    def set_train(self):
        self.unet_update.train()
        for n, _p in self.unet_update.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def forward(self, c_t, prompt=None, neg_prompt_tokens=None, prompt_tokens=None, deterministic=True, r=1.0, noise_map=None, args=None):

        caption_enc = self.text_encoder(prompt_tokens)[0]
        neg_caption_enc = self.text_encoder(neg_prompt_tokens)[0]

        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc.to(torch.float32),).sample
        x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample

        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image, caption_enc, neg_caption_enc

    def forward_latent(self, model, latents, timestep, prompt_embeds):
        
        noise_pred = model(
        latents,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        ).sample

        return noise_pred

    def compute_lora_loss(self, latents_pred, prompt_embeds, args):

        latents_pred = latents_pred.detach()
        prompt_embeds = prompt_embeds.detach()
        noise = torch.randn_like(latents_pred)
        bsz = latents_pred.shape[0]
        timesteps = torch.randint(0, self.sched.config.num_train_timesteps, (bsz,), device=latents_pred.device)
        timesteps = timesteps.long()
        noisy_latents = self.sched.add_noise(latents_pred, noise, timesteps)
        disc_pred = self.forward_latent(
            self.unet_update,
            timestep=timesteps,
            latents=noisy_latents,
            prompt_embeds=prompt_embeds
        )
        if args.snr_gamma_vsd is None:
            loss_d = F.mse_loss(disc_pred.float(), noise.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.sched, timesteps)
            if self.sched.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss_d = loss.mean()

        return loss_d

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def distribution_matching_loss(
        self,
        real_model,
        fake_model,
        noise_scheduler,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        args,
    ):
        bsz = latents.shape[0]
        min_dm_step = int(noise_scheduler.config.num_train_timesteps * args.min_dm_step_ratio)
        max_dm_step = int(noise_scheduler.config.num_train_timesteps * args.max_dm_step_ratio)

        timestep = torch.randint(min_dm_step, max_dm_step, (bsz,), device=latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timestep)

        with torch.no_grad():
            noise_pred = self.forward_latent(
                fake_model,
                latents=noisy_latents,
                timestep=timestep,
                prompt_embeds=prompt_embeds.float(),
            )
            pred_fake_latents = self.eps_to_mu(noise_scheduler, noise_pred, noisy_latents, timestep)

            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timestep_input = torch.cat([timestep] * 2)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

            noise_pred = self.forward_latent(
                real_model,
                latents=noisy_latents_input.to(dtype=torch.float16),
                timestep=timestep_input,
                prompt_embeds=prompt_embeds.to(dtype=torch.float16),
            )
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.cfg_vsd * (noise_pred_text - noise_pred_uncond)
            noise_pred.to(dtype=torch.float32)

            pred_real_latents = self.eps_to_mu(noise_scheduler, noise_pred, noisy_latents, timestep)

        weighting_factor = torch.abs(latents - pred_real_latents).mean(dim=[1, 2, 3], keepdim=True)

        grad = (pred_fake_latents - pred_real_latents) / weighting_factor
        loss = F.mse_loss(latents, self.stopgrad(latents - grad))
        return loss

    def stopgrad(self, x):
        return x.detach()

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_encoder_modules"], sd["unet_lora_decoder_modules"], sd["unet_lora_others_modules"] =\
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["rank_unet"] = self.lora_rank_unet
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k}
        torch.save(sd, outf)

class NAOSD(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").cuda()
        self.sched = make_1step_sched(args.pretrained_model_name_or_path)
        self.sched2 = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.args = args
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
        
        if args.pretrained_path is None:
            vae, lora_vae_modules_encoder, lora_vae_modules_decoder, lora_vae_others =\
                 initialize_vae(rank=args.lora_rank_vae, pretrained_model_name_or_path=args.pretrained_model_name_or_path, return_lora_module_names=True)
            unet, lora_unet_modules_encoder, lora_unet_modules_decoder, lora_unet_others =\
                 initialize_unet_sr(rank=args.lora_rank_unet, pretrained_model_name_or_path=args.pretrained_model_name_or_path, return_lora_module_names=True, args=args)
            self.lora_rank_unet = args.lora_rank_unet
            self.lora_rank_vae = args.lora_rank_vae
            self.lora_vae_modules_encoder, self.lora_vae_modules_decoder, self.lora_vae_others = \
                lora_vae_modules_encoder, lora_vae_modules_decoder, lora_vae_others
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = \
                lora_unet_modules_encoder, lora_unet_modules_decoder, lora_unet_others
        
        self.unet, self.vae = unet, vae
        
        if args.pretrained_path is not None:
            print('==================================> loading pre-trained weight')
            sd = torch.load(args.pretrained_path)
            self.load_ckpt_from_state_dict(sd)
            self.lora_rank_unet = sd['rank_unet']
            self.lora_rank_vae = sd['rank_vae']
            self.lora_vae_modules_encoder, self.lora_vae_modules_decoder, self.lora_vae_others = \
                sd['vae_lora_encoder_modules'], sd['vae_lora_decoder_modules'], sd['vae_lora_others_modules']
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = \
                sd['unet_lora_encoder_modules'], sd['unet_lora_decoder_modules'], sd['unet_lora_others_modules']

        self.unet, self.vae = self.unet.cuda(), self.vae.cuda()
        self.timesteps = torch.tensor([args.time_step], device="cuda").long()
        self.timestepsnoise = torch.tensor([args.time_step_noise], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def encode_prompt(self, prompt):
        with torch.no_grad():
            text_input_ids = self.tokenizer(
                prompt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            prompt_embeds = self.text_encoder(
                text_input_ids.to(self.text_encoder.device),
            )[0]
        return prompt_embeds

    def forward(self, c_t, positive_prompt=None, negative_prompt=None, args=None):
        caption_enc = self.encode_prompt(positive_prompt)
        neg_caption_enc = self.encode_prompt(negative_prompt)
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        noise = torch.randn_like(encoded_control)
        encoded_control = self.sched2.add_noise(encoded_control, noise, self.timestepsnoise)

        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc.to(torch.float32),).sample
        x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample
        output_image = output_image.clamp(-1, 1)

        return output_image, x_denoised, caption_enc, neg_caption_enc, noise

    def save_model(self, outf):
        sd = {}
        sd["vae_lora_encoder_modules"], sd["vae_lora_decoder_modules"], sd["vae_lora_others_modules"] =\
            self.lora_vae_modules_encoder, self.lora_vae_modules_decoder, self.lora_vae_others 
        sd["unet_lora_encoder_modules"], sd["unet_lora_decoder_modules"], sd["unet_lora_others_modules"] =\
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        torch.save(sd, outf)
    
    def load_ckpt_from_state_dict(self, sd):
        # load unet lora
        lora_conf_encoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(sd["state_dict_unet"][n])

        # load vae lora
        vae_lora_conf_encoder = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_encoder_modules"])
        vae_lora_conf_decoder = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_decoder_modules"])
        self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
        self.vae.add_adapter(vae_lora_conf_decoder, adapter_name="default_decoder")
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.data.copy_(sd["state_dict_vae"][n])

class GDPOSR(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").cuda()
        self.sched = make_1step_sched(args.pretrained_model_name_or_path)
        self.sched2 = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.args = args
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        vae = AutoencoderKL.from_pretrained(args.basemodel_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(args.basemodel_path, subfolder="unet")
        ref_unet = UNet2DConditionModel.from_pretrained(args.basemodel_path, subfolder="unet")
        
        if args.pretrained_path is None:
            print('==================================> randomly initiate the weight')
            unet, lora_unet_modules_encoder, lora_unet_modules_decoder, lora_unet_others =\
                 initialize_unet_sr(rank=args.lora_rank_unet, pretrained_model_name_or_path=args.basemodel_path, return_lora_module_names=True, args=args)
            self.lora_rank_unet = args.lora_rank_unet
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = \
                lora_unet_modules_encoder, lora_unet_modules_decoder, lora_unet_others
        
        self.unet, self.vae = unet, vae
        
        if args.pretrained_path is not None:
            print('==================================> loading pre-trained weight')
            sd = torch.load(args.pretrained_path)
            self.load_ckpt_from_state_dict(sd)
            self.lora_rank_unet = sd['rank_unet']
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = \
                sd['unet_lora_encoder_modules'], sd['unet_lora_decoder_modules'], sd['unet_lora_others_modules']

        self.unet, self.vae = self.unet.cuda(), self.vae.cuda()
        self.ref_unet = ref_unet.cuda()
        self.timesteps = torch.tensor([args.time_step], device="cuda").long()
        self.timestepsnoise = torch.tensor([args.time_step_noise], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.ref_unet.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.ref_unet.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        for n, _p in self.ref_unet.named_parameters():
                _p.requires_grad = False

    def encode_prompt(self, prompt):
        with torch.no_grad():
            text_input_ids = self.tokenizer(
                prompt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            prompt_embeds = self.text_encoder(
                text_input_ids.to(self.text_encoder.device),
            )[0]
        return prompt_embeds

    def forward(self, c_t, positive_prompt=[''], negative_prompt=[''], args=None):
        caption_enc = self.encode_prompt(positive_prompt)
        neg_caption_enc = self.encode_prompt(negative_prompt)
        with torch.no_grad():
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        encoded_control_ref = encoded_control
        noise = torch.randn_like(encoded_control)
        encoded_control = self.sched2.add_noise(encoded_control, noise, self.timestepsnoise)

        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc.to(torch.float32),).sample
        x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample
        output_image = output_image.clamp(-1, 1)

        with torch.no_grad():
            encoded_control_ref = self.sched2.add_noise(encoded_control_ref, noise, self.timestepsnoise)
            ref_model_pred = self.ref_unet(encoded_control_ref, self.timesteps, encoder_hidden_states=caption_enc.to(torch.float32),).sample
            ref_x_denoised = self.sched.step(ref_model_pred, self.timesteps, encoded_control_ref, return_dict=True).prev_sample
            ref_output_image = self.vae.decode(ref_x_denoised / self.vae.config.scaling_factor).sample
            ref_output_image = ref_output_image.clamp(-1, 1)

        return output_image, x_denoised, model_pred, caption_enc, neg_caption_enc, noise, ref_output_image, ref_x_denoised, ref_model_pred
    
    def GDPOReference(self, c_t, positive_prompt=[''], negative_prompt=[''], args=None, groupsize=6):
        
        with torch.no_grad():
            
            caption_enc = self.encode_prompt(positive_prompt).unsqueeze(1)
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            b,c,h,w=encoded_control.shape
            encoded_control = encoded_control.unsqueeze(1)
            caption_enc = caption_enc.repeat(1,groupsize,1,1)
            encoded_control = encoded_control.repeat(1, groupsize, 1, 1, 1)
            noise = torch.randn_like(encoded_control)
            output_image = torch.zeros_like(c_t).unsqueeze(1).repeat(1,groupsize,1,1,1)
            x_denoised = torch.zeros_like(noise)
            model_pred = torch.zeros_like(noise)
            for i in range(b):
                encoded_control_i = self.sched2.add_noise(encoded_control[i], noise[i], self.timestepsnoise)
                # print(encoded_control.shape, caption_enc.shape, self.timesteps.shape)
                model_pred_i = self.ref_unet(encoded_control_i, self.timesteps, encoder_hidden_states=caption_enc[i],).sample
                x_denoised_i = self.sched.step(model_pred_i, self.timesteps, encoded_control_i, return_dict=True).prev_sample
                output_image_i = self.vae.decode(x_denoised_i / self.vae.config.scaling_factor).sample
                output_image_i = output_image_i.clamp(-1, 1)
                output_image[i] = output_image_i
                x_denoised[i] = x_denoised_i
                model_pred[i] = model_pred_i

        return output_image, x_denoised, model_pred
    
    def save_model(self, outf):
        sd = {}
        sd["unet_lora_encoder_modules"], sd["unet_lora_decoder_modules"], sd["unet_lora_others_modules"] =\
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["rank_unet"] = self.lora_rank_unet
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        torch.save(sd, outf)
    
    def load_ckpt_from_state_dict(self, sd):
        # load unet lora
        lora_conf_encoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(sd["state_dict_unet"][n])

class GDPOSRTest(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").cuda()
        self.sched = make_1step_sched(args.pretrained_model_name_or_path)
        self.sched2 = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.args = args
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        vae = AutoencoderKL.from_pretrained(args.pretrained_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_path, subfolder="unet")
                
        self.unet, self.vae = unet, vae
        self.unet, self.vae = self.unet.cuda(), self.vae.cuda()
        self.timesteps = torch.tensor([args.time_step], device="cuda").long()
        self.timestepsnoise = torch.tensor([args.time_step_noise], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def encode_prompt(self, prompt):
        with torch.no_grad():
            text_input_ids = self.tokenizer(
                prompt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            prompt_embeds = self.text_encoder(
                text_input_ids.to(self.text_encoder.device),
            )[0]
        return prompt_embeds

    def forward(self, c_t, positive_prompt=['']):
        
        caption_enc = self.encode_prompt(positive_prompt)
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        noise = torch.randn_like(encoded_control)
        encoded_control = self.sched2.add_noise(encoded_control, noise, self.timestepsnoise)

        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc.to(torch.float32),).sample
        x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample
        output_image = output_image.clamp(-1, 1)
        

        return output_image
