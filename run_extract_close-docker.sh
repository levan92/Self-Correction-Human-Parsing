#!/bin/bash 

INDIR=${1}

EXTRACT_OUTDIR="${INDIR}-parse"
KERNEL=15
PARSE_OUTDIR="${EXTRACT_OUTDIR}_closed_k${KERNEL}"

EXTRACT_SCRIPT='/workspace/Self-Correction-Human-Parsing/simple_extractor-rglob.py'
CLOSE_SCRIPT='/workspace/Self-Correction-Human-Parsing/work-parse/close_masks.py'
MODEL='/workspace/Self-Correction-Human-Parsing/weights/exp-schp-201908261155-lip.pth'

python3 $EXTRACT_SCRIPT --dataset lip --model-restore $MODEL --gpu 0 --input-dir $INDIR --output-dir $EXTRACT_OUTDIR --logits --save_compressed

python3 ${CLOSE_SCRIPT} ${EXTRACT_OUTDIR} ${PARSE_OUTDIR} --kernel ${KERNEL}