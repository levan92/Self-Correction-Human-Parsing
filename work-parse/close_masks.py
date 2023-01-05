import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from random import randint, uniform, seed, shuffle
from tqdm import tqdm 
from color_transform import RGBTransform

# PARSE_ROOT="/home/acsguser/Data/WorkProgress/HandPose/data/handpose_dataset_parse"
# OUT_ROOT="/home/acsguser/Data/WorkProgress/HandPose/data/handpose_dataset_parse_closed_k15"

PARSE_ROOT="/home/acsguser/Data/WorkProgress/HandPose/data/annotated/20221219_fromTaeil-parse"
OUT_ROOT="/home/acsguser/Data/WorkProgress/HandPose/data/annotated/20221219_fromTaeil-parse_closed_k15"

CLOSING_KERNEL_SIZE=15

parse_out_path = Path(PARSE_ROOT)
out_path = Path(OUT_ROOT)
out_path.mkdir(exist_ok=True, parents=True)

npy_paths = [ f for f in parse_out_path.rglob('*.npy') ]
for npy_path in tqdm(npy_paths):
    parse_res = np.load(npy_path)
    subpath = npy_path.relative_to(parse_out_path)    

    mask = parse_res > 0
    mask_int = mask.astype(np.uint8)
    assert CLOSING_KERNEL_SIZE > 0 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(CLOSING_KERNEL_SIZE,CLOSING_KERNEL_SIZE))
    opened_mask = cv2.morphologyEx(mask_int,cv2.MORPH_CLOSE,kernel)

    out_npy_path = out_path / subpath
    out_npy_path.parent.mkdir(exist_ok=True, parents=True)
    np.save(str(out_npy_path), opened_mask)
