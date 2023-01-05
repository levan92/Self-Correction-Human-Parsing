#!/bin/bash 

# SCRIPT='/home/acsguser/Workspace/Self-Correction-Human-Parsing/simple_extractor.py'
# MODEL='/home/acsguser/Workspace/Self-Correction-Human-Parsing/weights/exp-schp-201908261155-lip.pth'
# # INDIR='/home/acsguser/Data/reid/mars/forClothestering'
# # OUTDIR='/home/acsguser/Data/reid/mars/forClothestering-out'

# INDIR='/home/acsguser/Data/WorkProgress/HandPose/data/handpose_dataset/hand'
# OUTDIR='/home/acsguser/Data/WorkProgress/HandPose/data/handpose_dataset_parse/hand'

# python3 $SCRIPT --dataset lip --model-restore $MODEL --gpu 0 --input-dir $INDIR --output-dir $OUTDIR --logits --save_compressed

# INDIR='/home/acsguser/Data/WorkProgress/HandPose/data/handpose_dataset/hand_class1'
# OUTDIR='/home/acsguser/Data/WorkProgress/HandPose/data/handpose_dataset_parse/hand_class1'

# python3 $SCRIPT --dataset lip --model-restore $MODEL --gpu 0 --input-dir $INDIR --output-dir $OUTDIR --logits --save_compressed

# INDIR='/home/acsguser/Data/WorkProgress/HandPose/data/handpose_dataset/hand_class4'
# OUTDIR='/home/acsguser/Data/WorkProgress/HandPose/data/handpose_dataset_parse/hand_class4'

# python3 $SCRIPT --dataset lip --model-restore $MODEL --gpu 0 --input-dir $INDIR --output-dir $OUTDIR --logits --save_compressed

# INDIR='/home/acsguser/Data/WorkProgress/HandPose/data/handpose_dataset/hand_class5'
# OUTDIR='/home/acsguser/Data/WorkProgress/HandPose/data/handpose_dataset_parse/hand_class5'

# python3 $SCRIPT --dataset lip --model-restore $MODEL --gpu 0 --input-dir $INDIR --output-dir $OUTDIR --logits --save_compressed


SCRIPT='/home/acsguser/Workspace/Self-Correction-Human-Parsing/simple_extractor-rglob.py'
MODEL='/home/acsguser/Workspace/Self-Correction-Human-Parsing/weights/exp-schp-201908261155-lip.pth'

INDIR='/home/acsguser/Data/WorkProgress/HandPose/data/annotated/20221219_fromTaeil'
OUTDIR='/home/acsguser/Data/WorkProgress/HandPose/data/annotated/20221219_fromTaeil-parse'

python3 $SCRIPT --dataset lip --model-restore $MODEL --gpu 0 --input-dir $INDIR --output-dir $OUTDIR --logits --save_compressed

