# Diversifying Detail and Appearance in Sketch-Based Face Image Synthesis

This code is our implementation of the following paper:

Takato Yoshikawa, Yuki Endo, Yoshihiro Kanamori: "Diversifying Detail and Appearance in Sketch-Based Face Image Synthesis" Visual Computer (Proc. of Computer Graphics Internatinal 2022), 2022. [[Project](http://www.cgg.cs.tsukuba.ac.jp/~yoshikawa/pub/sketch_to_diverse_image/)][[PDF (28 MB)](http://www.cgg.cs.tsukuba.ac.jp/~yoshikawa/pub/sketch_to_diverse_image/pdf/Yoshikawa_CGI2022.pdf)]

## Prerequisites
Run the following code to install all pip packages.

```
pip install -r requirements.txt
```

## Inference with our pre-trained models
1. Download our [pre-trained models]() for CelebA-HQ dataset and put them in the "pretrained_model" directory in the parent directory.
2. Run test.py
```
cd src
python test.py --input_name ../data/face_sketch/
```

## Citation
please cite our paper if you find the code useful:
```
@article{YoshikawaCGI22,
    author    = {Takato Yoshikawa and Yuki Endo and Yoshihiro Kanamori},
    title     = {Diversifying Detail and Appearance in Sketch-Based Face Image Synthesis},
    journal   = {Visual Computer (Proc. of Computer Graphics Internatinal 2022)},
    volume    = {},
    number    = {},
    pages     = {},
    year      = {2022}
}
```

## Acknowledgements
This code heavily borrows from the [DeepPS](https://github.com/VITA-Group/DeepPS) repository.