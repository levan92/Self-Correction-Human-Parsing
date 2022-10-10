from pathlib import Path
import random 
from shutil import copy

MARS_ROOT="/home/acsguser/Data/reid/mars"
wanted_sets = ['bbox_test', 'bbox_train']
SAMPLE = 3
OUTNAME = 'forClothestering'
SEED = 88

random.seed(SEED)
mars_root_path = Path(MARS_ROOT)
out_path = mars_root_path / OUTNAME
out_path.mkdir(exist_ok=True, parents=True)
count_img = 0 
pids = []
for set_ in wanted_sets: 
    set_path = mars_root_path / set_    
    for pid_path in set_path.glob('*'):
        if pid_path.is_dir(): 
            pid = pid_path.stem
            assert pid not in pids
            pids.append(pid)
            jpgs = [ f for f in pid_path.glob('*.jpg') ]
            for selected in random.sample(jpgs, k=SAMPLE): 
                print(f'Copied {selected} to {out_path}')
                copy(selected, out_path)
                count_img += 1

print(f'Total PIDs: {len(pids)}')
print(f'Copied out total {count_img} images sampling {SAMPLE} per pid.')