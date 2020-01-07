# porous_segmentation

This is an implementation of paper titled "Automatic Segmentation for Synchrotron Based Imaging of Porous Bread Dough".

## Getting Started

## Install
```
$ git clone https://github.com/salahali2019/porous_segmentation.git
$ cd porous_segmentation
$ pip install -r requirements.txt
$ pip install --upgrade tensorflow
$ pip install opencv-python
$ pip install keras
```

## Training on Your Own Dataset
```
$ python3 synthetic_generator.py overlapping_spheres --porosity=0.6  --size=23 --fileName='first' --three_3D_dir='synthetic/three_3D_dir' --grayscale_image_dir="synthetic/input" --GT_image_dir="synthetic/output"
$ python3 main.py train  --epoch=11 --BATCH_SIZE=8 --LR=0.08 --train_dir='synthetic' --valid_dir='validation' --test_dir='test --model_dir='model_weight.h5'
$ python3 main.py predict  --epoch=11 --BATCH_SIZE=8 --LR=0.08 --train_dir='synthetic' --valid_dir='real' --test_dir='test',--model_dir='model_weight.h5'

```
## Citation
