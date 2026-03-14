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

import sys
sys.path.append("./")
from diffusermodels.autoencoder_kl import AutoencoderKL as AutoencoderKLMerge
from diffusermodels.unet_2d_condition import UNet2DConditionModel as UNet2DConditionModelMerge


def UNetMergeLoRA(basemodel_path='', trainedmodel_path='', savepath='', savename=''):

    loraweight = torch.load(trainedmodel_path)
    # vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModelMerge.from_pretrained(basemodel_path, subfolder="unet")

    # load unet lora
    lora_conf_encoder = LoraConfig(r=loraweight["rank_unet"], init_lora_weights="gaussian", target_modules=loraweight["unet_lora_encoder_modules"])
    lora_conf_decoder = LoraConfig(r=loraweight["rank_unet"], init_lora_weights="gaussian", target_modules=loraweight["unet_lora_decoder_modules"])
    lora_conf_others = LoraConfig(r=loraweight["rank_unet"], init_lora_weights="gaussian", target_modules=loraweight["unet_lora_others_modules"])
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")
    for n, p in unet.named_parameters():
        if "lora" in n or "conv_in" in n:
            p.data.copy_(loraweight["state_dict_unet"][n])

    unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])
    unet = unet.merge_and_unload()
    unet.save_pretrained(os.path.join(savepath, savename))

def VAEMergeLoRA(basemodel_path='', trainedmodel_path='', savepath='', savename=''):

    loraweight = torch.load(trainedmodel_path)
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")

    # load vae lora
    vae_lora_conf_encoder = LoraConfig(r=loraweight["rank_vae"], init_lora_weights="gaussian", target_modules=loraweight["vae_lora_encoder_modules"])
    vae_lora_conf_decoder = LoraConfig(r=loraweight["rank_vae"], init_lora_weights="gaussian", target_modules=loraweight["vae_lora_decoder_modules"])
    vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
    vae.add_adapter(vae_lora_conf_decoder, adapter_name="default_decoder")
    for n, p in vae.named_parameters():
        if "lora" in n:
            p.data.copy_(loraweight["state_dict_vae"][n])

    vae.set_adapter(['default_encoder'])
    vae = vae.merge_and_unload()
    vae.save_pretrained(os.path.join(savepath, savename))

unetbasemodel_path=''
unettrainedmodel_path=''
unetsavepath=''
unetsavename=''
UNetMergeLoRA(unetbasemodel_path, unettrainedmodel_path, unetsavepath, unetsavename)

vaebasemodel_path=''
vaetrainedmodel_path=''
vaesavepath=''
vaesavename=''
VAEMergeLoRA(vaebasemodel_path, vaetrainedmodel_path, vaesavepath, vaesavename)