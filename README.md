<div align="center">
<h2>GDPO-SR: Group Direct Preference Optimization for One-Step Generative Image Super-Resolution</h2>

[Qiaosi Yi](https://dblp.org/pid/249/8335.html)<sup>1,2</sup>
| [Shuai Li](https://scholar.google.com/citations?hl=zh-CN&user=Bd73ldQAAAAJ)<sup>1</sup>
| [Rongyuan Wu](https://scholar.google.com/citations?user=A-U8zE8AAAAJ&hl=zh-CN)<sup>1,2</sup>
| [Lingchen Sun](https://scholar.google.com/citations?hl=zh-CN&tzom=-480&user=ZCDjTn8AAAAJ)<sup>1,2</sup>
| [Zhengqiang zhang](https://scholar.google.com/citations?user=UX26wSMAAAAJ&hl=en)<sup>1,2</sup>
| [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang)<sup>1,2</sup>

<sup>1</sup>The Hong Kong Polytechnic University, <sup>2</sup>OPPO Research Institute
</div>

[![](https://img.shields.io/badge/ArXiv%20-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/pdf/2603.16769)&nbsp; [![weights](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-model%20weights-blue)](https://huggingface.co/Joypop/GDPO/tree/main)&nbsp;  CVPR26


## ⏰ Update
- **2026.3.19: Paper is released on [ArXiv](https://arxiv.org/pdf/2603.16769).
- **2026.3.12**: The training code and testing code are released.
- **2026.3.10**: The repo is released.


## ⚙ Dependencies and Installation
```shell
## git clone this repository
git clone https://github.com/Joyies/GDPO.git
cd GDPO
# create an environment
conda create -n GDPO python=3.10
conda activate GDPO
pip install --upgrade pip
pip install -r requirements.txt
```

## 🏂 Quick Inference

#### Step 1: Download the pretrained models
- Download the model weights [![weights](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-model%20weights-blue)](https://huggingface.co/Joypop/GDPO/tree/main/ckp)&nbsp; and put the model weights in the `ckp/`:

#### Step 2: Prepare testing data and run testing command 
You can modify input_path and output_path to run testing command. The input_path is the path of the test image and the output_path is the path where the output images are saved.
```shell
CUDA_VISIBLE_DEVICES=0, python GDPOSR/inferences/test.py \
--input_path test_LR \
--output_path experiment/GDPOSR \
--pretrained_path ckp/GDPOSR \
--pretrained_model_name_or_path stable-diffusion-2-1-base \
--ram_ft_path ckp/DAPE.pth \
--negprompt 'dotted, noise, blur, lowres, smooth' \
--prompt 'clean, high-resolution, 8k' \
--upscale 1 \
--time_step=100 \
--time_step_noise=250 
```
or 
```shell
bash scripts/test/test.sh
```

## 🚄 Training Phase

### Step1: Prepare training data
Download the [LSIDR dataset](https://github.com/ofsoundof/LSDIR) and [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset) and crop multiple 512×512 image patches using a sliding window with a stride of 64 pixels;


### Step2: Train NAOSD.
```shell
bash scripts/train/train_NAOSD.sh
```
The hyperparameters in train_NAOSD.sh can be modified to suit different experimental settings. Besides, after training with NAOSD, you can use ```GDPOSR/mergelora.py``` to merge the LoRA into the UNet and VAE as base model for subsequent reinforcement learning training and inference.
### Step3: Train GDPO-SR
```shell
bash scripts/train/train_GDPOSR.sh
```
The hyperparameters in train_GDPOSR.sh can be modified to suit different experimental settings. Besides, after training with GDPO-SR, you can use ```GDPOSR/mergelora.py``` to merge the LoRA into the UNet for subsequent inference.
## 🔗 Citations
```
@article{yi2026gdpo,
  title={GDPO-SR: Group Direct Preference Optimization for One-Step Generative Image Super-Resolution},
  author={Yi, Qiaosi and Li, Shuai and Wu, Rongyuan and Sun, Lingchen and Zhang, Zhengqiang and Zhang, Lei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
## ©️ License
This project is released under the [Apache 2.0 license](LICENSE).
## 📧 Contact
If you have any questions, please contact: qiaosiyijoyies@gmail.com
