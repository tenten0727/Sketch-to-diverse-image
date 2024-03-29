# Diversifying Detail and Appearance in Sketch-Based Face Image Synthesis

![teaser](docs/teaser.png)

This code is our implementation of the following paper:

Takato Yoshikawa, Yuki Endo, Yoshihiro Kanamori: "Diversifying Detail and Appearance in Sketch-Based Face Image Synthesis" The Visual Computer (Proc. of Computer Graphics Internatinal 2022), 2022. [[Project](http://www.cgg.cs.tsukuba.ac.jp/~yoshikawa/pub/sketch_to_diverse_image/)][[PDF (28 MB)](http://www.cgg.cs.tsukuba.ac.jp/~yoshikawa/pub/sketch_to_diverse_image/pdf/Yoshikawa_CGI2022.pdf)]

## Prerequisites
Run the following code to install all pip packages.

```
pip install -r requirements.txt
```

## Inference with our pre-trained models
1. Download our [pre-trained models](https://drive.google.com/file/d/1OMMnm5Ez5rq1YYbbLWTelpL4EKwhzjy0/view?usp=sharing) for the CelebA-HQ dataset and put them into the "pretrained_model" directory in the parent directory.
2. Download the zip file from "Human-Drawn Facial sketches" in [DeepPS](https://github.com/VITA-Group/DeepPS), unzip it, and put the "sketches" directory into the "data" directory in the parent directory.
3. Run test.py
```
cd src
python test.py
```

## Training
1. Download the edge map dataset from [Google Drive](https://drive.google.com/drive/folders/1NSuh0L5RTFQq0lwZq0NRAiloX_ar-kWa?usp=sharing) to the "data" directory.
2. Download the [CelebA-HQ dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html) and run resize_image.py to resize the image.
```
cd src
python resize_image.py --input path/to/CelebA-HQ/dataset --output ../data/CelebA-HQ256
```

### Training the detail network H
```
python train.py \
--train_path ../data/CelebA-HQ256_DFE \
--edge_path ../data/CelebA-HQ256_HED \
--edgeSmooth \
--save_model_name network-H
```
### Training the appearance network F
```
python train.py \
--train_path ../data/CelebA-HQ256 \
--edge_path ../data/CelebA-HQ256_DFE \
--weight_feat 0.0 \
--save_model_name network-F
```

## Citation
Please cite our paper if you find the code useful:
```
@article{YoshikawaCGI22,
    author    = {Takato Yoshikawa and Yuki Endo and Yoshihiro Kanamori},
    title     = {Diversifying Detail and Appearance in Sketch-Based Face Image Synthesis},
    journal   = {The Visual Computer (Proc. of Computer Graphics Internatinal 2022)},
    volume    = {38},
    number    = {9},
    pages     = {3121-–3133},
    year      = {2022}
}
```

## Acknowledgements
This code heavily borrows from the [DeepPS](https://github.com/VITA-Group/DeepPS) repository.
