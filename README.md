# SupFAD (Train once and test other)
>  [**SupFAD: Superordinary FAD with Dynamic Object-Agnostic Prompt Learning for Zero-Shot Fabric Anomaly Detection**]
>
> by XinYing Li*, Tong Wu.


## Updates

- **02.10.2025**: The project has already been uploaded.
- **XX.XX.2025**: Code has been released !!!

## Introduction 


## Framework of SupFAD



## Analysis of different text prompt templates



## How to Run
### Prepare your dataset
Download the dataset below:

* Fabric Domain:
The datasets generated during the current study are not publicly available due Self-built datasets and limited by industry, but are available from the corresponding author on reasonable request.
* Industrial Domain:
[MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad), [VisA](https://github.com/amazon-science/spot-diff)




### Generate the dataset JSON
Take MVTec AD for example (With multiple anomaly categories)

Structure of MVTec Folder:
```
mvtec/
│
├── meta.json
│
├── bottle/
│   ├── ground_truth/
│   │   ├── broken_large/
│   │   │   └── 000_mask.png
|   |   |   └── ...
│   │   └── ...
│   └── test/
│   │   ├── broken_large/
│   │   │   └── 000.png
|   │   |   └── ...
│   │   └── ...
│   └── train/
│       ├── good/
│       │   └── 000.png
|       |   └── ...
└── ...
```

```bash
cd generate_dataset_json
python mvtec.py
```


```
Select the corresponding script and run it (we provide all scripts for datasets that SupFAD reported). The generated JSON stores all the information that SupFAD needs. 

### Custom dataset (optional)
1. Create a new JSON script in fold [generate_dataset_json](https://github.com/zqhang/AnomalyCLIP/tree/main/generate_dataset_json) according to the fold structure of your own datasets.
2. Add the related info of your dataset (i.e., dataset name and class names) in script [dataset\.py](https://github.com/zqhang/AnomalyCLIP/blob/main/dataset.py)

### Run SupFAD
* Quick start (use the pre-trained weights)
```bash
bash test.sh
```
  
* Train your own weights
```bash
bash train.sh
```


## Main results (We test fabric datasets by training once on MVTec AD. For MVTec AD, SupFAD is trained on fabric.)

### Fabric dataset
### Industrial dataset

## Visualization



## We provide the reproduction of WinCLIP [here](https://github.com/zqhang/WinCLIP-pytorch)


* We thank for the code repository: [open_clip](https://github.com/mlfoundations/open_clip), [DualCoOp](https://github.com/sunxm2357/DualCoOp), [CLIP_Surgery](https://github.com/xmed-lab/CLIP_Surgery), and [VAND](https://github.com/ByChelsea/VAND-APRIL-GAN/tree/master).

## BibTex Citation

If you find this paper and repository useful, please cite our paper.

```

```
