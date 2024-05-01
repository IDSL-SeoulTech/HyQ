# HyQ: Hardware-Friendly Post-Training-Quantization for CNN-Transformer Hybrid Networks
HyQ is a set of techniques designed for quantizing CNN-transformer hybrid models in a hardware-friendly way. This repository contains the official source code used to run experiments for the paper.

![image](https://github.com/IDSL-SeoulTech/HyQ/assets/50408754/739c8c07-0f8d-4b03-aaa2-5b059fb23f0d)

## Abstract
Hybrid models that combine convolutional neural networks (CNNs) and vision transformers (ViTs) have recently emerged as state-of-the-art computer vision models. To efficiently deploy these hybrid models involving significant amounts of parameters on resource-constrained mobile/edge devices, quantization is emerging as a promising solution. However, post-training quantization (PTQ), which does not require retraining or labeled data, has not been extensively studied for hybrid models. 
%Furthermore, owing to the complex structure of the hybrid model that combines both convolutions and transformer blocks, quantizing hybrid models using existing methods results in significant accuracy drops. 
In this study, we propose a novel PTQ technique specialized for CNN-transformer hybrid models by considering the hardware design of hybrid models on AI accelerators such as GPUs and FPGAs. First, we introduce quantization-aware distribution scaling (QADS) to address the large outliers caused by inter-channel variance in convolution layers. Furthermore, in the transformer block, we propose approximating the integer-only softmax with a linear function. This approach allows us to avoid costly FP32/INT32 multiplications, resulting in more efficient computations. We demonstrate the superiority of our proposed method on the ImageNet-1k dataset using various hybrid models. In particular, the proposed quantization method with INT8 precision demonstrated a 0.39\% accuracy drop compared with the FP32 baseline on MobileViT-s. Furthermore, when implemented on the FPGA platform, the proposed linear softmax achieved significant resource savings, reducing the look-up table (LUT) and flip-flop (FF) usage by $1.8 \sim 2.1\times$ and $1.3 \sim 1.9\times$, respectively, compared with the existing second-order polynomial approximation.

## Getting Started

### Install

- Create a conda virtual environment and activate it.

```bash
conda create -n HyQ python=3.10 -y
conda activate HyQ
```

- Install PyTorch and torchvision. e.g.,

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch 
or
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

pip install timm==0.9.5
```

### Data preparation

You should download the standard ImageNet Dataset.

```
├── imagenet
│   ├── train
│
│   ├── val
```

### Running the Experiment
To run the experiment, use the following command:

```
cd HyQ
python test_quant.py mobilevit_xxs <dataset folder> --quant --QADS --ptf --lis
```
Replace <dataset folder> with the path to your dataset.

- `mobilevit_xxs`: model architecture, which can be replaced by `mobilevit_xxs`, `mobilevit_xs`, and `mobilevit_s`.

- `--quant`: whether to quantize the model.

- `--QADS`: whether to use **Quantization-Aware Distribution Scaling**

- `--ptf`: whether to use Power-of-Two Factor Integer Layernorm.

- `--lis`: whether to use Log-Integer-Softmax.

- `--quant-method`: quantization methods of activations, which can be chosen from `minmax`, `ema`, `percentile` and `omse`.


## Experiments

### MobileViTv1
<center>
| Model | Method | Prec. (W/A) | Size (MB) | Top-1 Acc. (%) | Acc. Drop (%) |
|:-----:|:------:|:-----------:|:---------:|:---------------:|:-------------:|
| MobileViT-xxs | Baseline | 32/32 | 1.27M | 68.94 | - |
| | Ours | 8/8 | 0.32M | 68.15 | 0.79 |
| MobileViT-xs | Baseline | 32/32 | 2.32M | 74.63 | - |
| | Ours | 8/8 | 0.58M | 73.99 | 0.64 |
| MobileViT-s | Baseline | 32/32 | 5.58M | 78.32 | - |
| | Ours | 8/8 | 1.40M | 77.93 | 0.39 |
</center>

### EfficientFormer and MobileViTv2
| Model | Method | Prec. (W/A) | Top-1 Acc. (%) | Acc. Drop (%) |
|-------|--------|-------------|-----------------|---------------|
| EfficientFormer-L1 | Baseline | 32/32 | 80.50 | - |
| | Ours | 8/8 | 78.55 | 1.95 |
| EfficientFormer-L3 | Baseline | 32/32 | 82.55 | - |
| | Ours | 8/8 | 82.26 | 0.29  |
| EfficientFormer-L7 | Baseline  | 32/32 | 83.38 | -         |
| | Ours      | 8/8   | 82.66 |  0.72     |
| MobileViTv2-50 | Baseline  | 32/32 | 70.16 | -         |
| | Ours      | 8/8   | 69.16 | 1.00      |
| MobileViTv2-75 | Baseline | 32/32 | 75.61 | -         |
| | Ours      | 8/8   | 74.47 | 1.14      |



## Contributing

Our code is based on [FQ-ViT](https://github.com/megvii-research/FQ-ViT). We sincerely thank the contributors of FQ-ViT for their outstanding work.
