# NTIRE2021_compressed_video_enhancement
Code of the DUVE network for NTIRE 2021 Quality enhancement of heavily compressed videos - Track 3 Fixed bit-rate

### [Paper](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Yang_NTIRE_2021_Challenge_on_Quality_Enhancement_of_Compressed_Video_Methods_CVPRW_2021_paper.pdf)

## Dependencies
cuda:10.0 pytorch:1.2 python:3.6 (Anaconda is recommended)

## The Link to the pretrained model
https://drive.google.com/file/d/1cyJpf32SrAZzB1cdgZ0FJtC1ugjKBJ_a/view?usp=sharing

Download the pretrained model, and then put it in the ./pretrained_model/weight.

## Test
cd /code

python test.py

## Acknowledgement
Our code is based on : [RRN](https://github.com/junpan19/RRN) and [DRN](https://github.com/guoyongcs/DRN)
