# NTIRE2021_compressed_video_enhancement
Code of the DUVE network for NTIRE 2021 Quality enhancement of heavily compressed videos - Track 3 Fixed bit-rate

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
