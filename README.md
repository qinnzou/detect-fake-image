# detect-fake-image
This is the source code of metric learning for anti-compression facial forgery detection. We provide the codes and the trained models.

## Data Preprocess
We experiment on [FaceForensics++](https://github.com/ondyari/FaceForensics). Data partitioning can be found in `jsons/`. We use dlib to extract faces and facial locations of frames are stored in `jsons/`. Use `create_faces_from_dlib_locs.py` to extract faces of every compression level and the corresponding mask area. The dataset folder should be 
```
face
├── Deepfakes
│   ├── c23
│   │   ├── 000_003
│   │   │   ├── 105.png
│   │   │   ├── 106.png
│   │   │   ├── ...
│   │   │   ├── ...
│   │   ├── 001_870
│   │   ├── ...
│   │   ├── ...
│   ├── c40
│   └── masks
├── Face2Face
│   ├── c23
│   ├── c40
│   └── masks
├── FaceSwap
│   ├── c23
│   ├── c40
│   └── masks
├── NeuralTextures
│   ├── c23
│   ├── c40
│   └── masks
└── real
    ├── c23
    └── c40
```

## Pretrained Models
Pretrained models using PyTorch are available using the link below.
Baidu Drive: https://pan.baidu.com/s/1nfvPF-mSkD3FC8whpSpo8Q 
passcodes: 1234


## Set up
To evlauate the performance of a pre-trained model, you should put the pretrained model released by us into `./checkpoints/` and  and change  `DATASET` in `test.py` and `eval.py` to select the test dataset. And then run
```
python test.py
python eval.py
```

## Reference
Some of the codes are based on https://github.com/JStehouwer/FFD_CVPR2020 and https://github.com/ondyari/FaceForensics

## Citation
```
@inproceedings{cao2021metric,
  title={Metric Learning for Anti-Compression Facial Forgery Detection},
  author={Cao, Shenhao and Zou, Qin and Mao, Xiuqing and Ye, Dengpan and Wang, Zhongyuan},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia (ACM MM 2021)},
  pages={1929--1937},
  year={2021}
}

If you use the codes in our work, you should also cite the following papers:

@inproceedings{cvpr2020-dang,
  title={On the Detection of Digital Face Manipulation},
  author={Hao Dang, Feng Liu, Joel Stehouwer, Xiaoming Liu, Anil Jain},
  booktitle={In Proceeding of IEEE Computer Vision and Pattern Recognition (CVPR 2020)},
  address={Seattle, WA},
  year={2020}
}

@inproceedings{roessler2019faceforensicspp,
	author = {Andreas R\"ossler and Davide Cozzolino and Luisa Verdoliva and Christian Riess and Justus Thies and Matthias Nie{\ss}ner},
	title = {Face{F}orensics++: Learning to Detect Manipulated Facial Images},
	booktitle= {International Conference on Computer Vision (ICCV)},
	year = {2019}
}
```
