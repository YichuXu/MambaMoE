

<p align="center">

  <h2 align="center"><strong>MambaMoE: Mixture-of-Spectral-Spatial-Experts State Space Model for Hyperspectral Image Classification</strong></h2>

<div align="center">
<h5>

<em>Yichu Xu<sup>1</sup>, Di Wang<sup>1 </sup>, Hongzan Jiao<sup>1</sup>, Lefei Zhang<sup>1 †</sup>, Liangpei Zhang<sup>1,2</sup> </em>
    <br><br>
       	<sup>1</sup> Wuhan University, China, <sup>2</sup> Henan Academy of Sciences, China, <sup>†</sup>Corresponding author <br> 
    </h5>
[![IF paper](https://img.shields.io/badge/IF-paper-00629B.svg)](https://www.sciencedirect.com/science/article/pii/S089360802500190X?dgcid=author1) [![arXiv paper](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2504.20509)
</div>



## 📖Overview

* [**MambaMoE**](https://arxiv.org/abs/2504.20509) is a novel Mamba-based
spectral-spatial mixture-of-experts framework for HSI
classification. To the best of our knowledge, this is the
first MoE-based deep network introduced in the HSI
classification domain, enabling adaptive extraction of
spectral-spatial joint features tailored to the diverse
characteristics of land covers.  

<div align="center">
  <img src="./figures/MambaMoE.png"><br><br>
</div>

## 🚀Let's Get Started!
### `A. Installation`
**Step 1: Clone the repository:**

Clone this repository and navigate to the project directory:
```bash
git clone https://github.com/YichuXu/MambaMoE.git
cd MambaMoE
```

**Step 2: Environment Setup:**

It is recommended to set up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:

***Create and activate a new conda environment***

```bash
conda create -n MambaMoE
conda activate MambaMoE
```

***Install dependencies***

Our method uses python 3.8, pytorch 1.13, other environments are in requirements.txt
```bash
pip install -r requirements.txt
```

### `B. Data Preparation`

Download HSI classification dataset from [Google Drive](https://drive.google.com/drive/folders/1iPFLdrAFUveqwCtMpf5859pQhGXN_z4J?usp=drive_link) or [Baidu Drive (百度网盘)](https://pan.baidu.com/s/1bSqq-Uv3AC5qfRmqxbMjfg?pwd=2025) and put it under the [dataset] folder. It will have the following structure: 
```
${DATASET_ROOT}   # Dataset root directory
├── datasets
│   │
│   ├── pu        # Pavia University data
│   │   ├──PaviaU.mat
│   │   ├──PaviaU_gt.mat
│   │
│   ├── houston13  # Houston 2013 data
│   │   ├──GRSS2013.mat
│   │   ├──GRSS2013_gt.mat 
│   │     
│   ├── whuhc     # Whu-HanChuan data
│   │   ├──WHU_Hi_HanChuan.mat
│   │   ├──WHU_Hi_HanChuan_gt.mat 
│   │
│   ├── other HSI Datasets   
│   │   ├ ... 
│   │    

```

### `C. Performance Evaluation`
- The following commands show how to train and evaluate DSFormer for HSI classification:
```bash
python main.py --model DSFormer --dataset_name pu --num_run 10 --epoch 500 --device 0 --dataID 1 --patch_size 10 --k 2/5 --train_num 30 --group_num 4 --ps 2
python main.py --model DSFormer --dataset_name ip --num_run 10 --epoch 500 --device 1 --dataID 4 --patch_size 10 --k 4/5 --train_num 50 --group_num 4 --ps 2
python main.py --model DSFormer --dataset_name houston13 --num_run 10 --epoch 500 --device 2 --dataID 3 --patch_size 10 --k 3/5 --train_num 50 --group_num 4 --ps 2
python main.py --model DSFormer --dataset_name whuhh --num_run 10 --epoch 500 --device 3 --dataID 7 --patch_size 10 --k 3/5 --train_num 50 --group_num 4 --ps 2
```

## 📜Reference

if you find it useful for your research, please consider giving this repo a ⭐ and citing our paper! We appreciate your support！😊
```
@ARTICLE{Xu2025MambaMoE,
  author={Xu, Yichu and Wang, Di and Jiao, Hongzan Zhang, Lefei and Zhang, Liangpei},
  title={MambaMoE: Mixture-of-Spectral-Spatial-Experts State Space Model for Hyperspectral Image Classification}, 
  journal={Information Fusion},
  volume = {},
  pages = {},
  year = {2025}
}
```

## 🙋Q & A
**For any questions, please [contact us.](mailto:xuyichu@whu.edu.cn)**




