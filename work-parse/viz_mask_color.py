import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from random import randint, uniform, seed, shuffle

from color_transform import RGBTransform

PARSE_CLASSES = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', 'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
WANTED_CLASSES = None
IMG_ROOT="/home/acsguser/Data/WorkProgress/HandPose/data/handpose_dataset"
PARSE_OUT_ROOT="/home/acsguser/Data/WorkProgress/HandPose/data/handpose_dataset_parse"
OUT_ROOT="/home/acsguser/Data/WorkProgress/HandPose/data/handpose_segm_color"
SEED=88
CLOSING_KERNEL_SIZE=15

img_root_path = Path(IMG_ROOT)
if WANTED_CLASSES is None: 
    WANTED_CLASSES = PARSE_CLASSES
wanted_classes_idxes = [ PARSE_CLASSES.index(cl) for cl in WANTED_CLASSES ]
parse_out_path = Path(PARSE_OUT_ROOT)
out_path = Path(OUT_ROOT)
out_path.mkdir(exist_ok=True, parents=True)

seed(SEED)

ctrans = RGBTransform()

npy_paths = [ f for f in parse_out_path.rglob('*.npy') ]
shuffle(npy_paths)

for npy_path in npy_paths:
    parse_res = np.load(npy_path)
    subpath = npy_path.relative_to(parse_out_path)    
    subpath = subpath.parent / f"{subpath.stem}.jpg"
    img_path = img_root_path / subpath
    assert img_path.is_file()
    img = cv2.imread(str(img_path))

    img_color = img.copy()
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img = pil_img.convert('RGB') # ensure image has 3 channels
    trans_img = ctrans.mix_with((randint(0,255),randint(0,255),randint(0,255)), factor=uniform(0.25, 0.85)).applied_to(pil_img)
    trans_img_np = np.array(trans_img)
    trans_img_bgr = cv2.cvtColor(trans_img_np, cv2.COLOR_RGB2BGR)

    mask = parse_res > 0
    mask_int = mask.astype(np.uint8)
    if CLOSING_KERNEL_SIZE > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(CLOSING_KERNEL_SIZE,CLOSING_KERNEL_SIZE))
        opened_mask = cv2.morphologyEx(mask_int,cv2.MORPH_CLOSE,kernel)
        viz_close = np.concatenate([mask_int, opened_mask], axis=1)
    else:
        opened_mask = mask_int

    img[opened_mask==1] = trans_img_bgr[opened_mask==1]

    out_img_path = out_path / subpath
    print(out_img_path)
    out_img_path.parent.mkdir(exist_ok=True, parents=True)

    cv2.imwrite(str(out_img_path), img)

    #     viz_img = np.concatenate((wanted_img, strip), axis=1)

    #     viz_img_out_path = out_path / f'{fn}-color.jpg'
    #     cv2.imwrite(str(viz_img_out_path), viz_img)

    #     pal_out_path = out_path / f'{fn}-color.npy'
    #     np.save(str(pal_out_path), palette)

    # # import ipdb; ipdb.set_trace()