import numpy as np 
import cv2
from pathlib import Path
from tqdm import tqdm 

from colorthief import MMCQ 

PARSE_CLASSES = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', 'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
WANTED_CLASSES = ['Upper-clothes', 'Dress', 'Coat',]
IMG_ROOT="/home/acsguser/Data/reid/mars/forClothestering"
PARSE_OUT_ROOT="/home/acsguser/Data/reid/mars/forClothestering-out"
OUT_ROOT="/home/acsguser/Data/reid/mars/forClothestering-tops-out"
COLOR_COUNT=3

img_root_path = Path(IMG_ROOT)
wanted_classes_idxes = [ PARSE_CLASSES.index(cl) for cl in WANTED_CLASSES ]
parse_out_path = Path(PARSE_OUT_ROOT)
out_path = Path(OUT_ROOT)
out_path.mkdir(exist_ok=True, parents=True)

npy_files = [f for f in parse_out_path.glob('*.npy')]

for npy_file in tqdm(npy_files):
    file_path = Path(npy_file)
    fn = file_path.stem
    imgpath = img_root_path / f'{fn}.jpg'
    assert imgpath.is_file()
    img = cv2.imread(str(imgpath))

    logits = np.load(npy_file)
    parse = np.argmax(logits, axis=2)
    mask = np.isin(parse, wanted_classes_idxes)

    wanted_img = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))

    img_out_path = out_path / f'{fn}.png'
    mask_out_path = out_path / f'{fn}-mask.npy'

    cv2.imwrite(str(img_out_path), wanted_img)
    np.save(str(mask_out_path), mask)

    valid_pixels = []

    amal_img = np.append(wanted_img, mask.astype(wanted_img.dtype)[...,None], axis=2)
    h,w,c = amal_img.shape
    for pix in amal_img.reshape((h*w,c)): 
        r,g,b,want = pix
        if want:
            if not (r > 250 and g > 250 and b > 250):
                valid_pixels.append((r, g, b))
    
    if len(valid_pixels)>0:
        cmap = MMCQ.quantize(valid_pixels, COLOR_COUNT)
        palette = cmap.palette

        strip = np.zeros((h,20,3))
        unit = h//len(palette)
        for i, pal in enumerate(palette):
            strip[i*unit:(i+1)*unit] += pal 

        viz_img = np.concatenate((wanted_img, strip), axis=1)

        viz_img_out_path = out_path / f'{fn}-color.jpg'
        cv2.imwrite(str(viz_img_out_path), viz_img)

        pal_out_path = out_path / f'{fn}-color.npy'
        np.save(str(pal_out_path), palette)

    # import ipdb; ipdb.set_trace()