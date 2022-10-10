#!/bin/bash 

SCRIPT='/home/acsguser/Workspace/Self-Correction-Human-Parsing/simple_extractor.py'
MODEL='/home/acsguser/Workspace/Self-Correction-Human-Parsing/weights/exp-schp-201908261155-lip.pth'
INDIR='/home/acsguser/Data/reid/mars/forClothestering'
OUTDIR='/home/acsguser/Data/reid/mars/forClothestering-out'

python3 $SCRIPT --dataset lip --model-restore $MODEL --gpu 0 --input-dir $INDIR --output-dir $OUTDIR --logits 