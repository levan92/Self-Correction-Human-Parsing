from pathlib import Path
from tqdm import tqdm
from colorsys import rgb_to_hsv, hsv_to_rgb
import math 

import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

ROOT="/home/acsguser/Data/reid/mars/forClothestering-tops-out/"
K = 3
SEED = 88
VAR_THRESH = 20
NUM_TOP_COLORS = 3

root_path = Path(ROOT)

color_npys = [ f for f in root_path.glob('*-color.npy') ]

rgb_colors = []
hsv_colors = []
stds = []
ids = []
for color_npy in tqdm(color_npys): 
    name = color_npy.stem[:4]
    ids.extend([ name for _ in range(NUM_TOP_COLORS) ])
    palette = np.load(color_npy)
    stds.extend( [ np.std(c) for c in palette[:NUM_TOP_COLORS] ] )
    rgb_colors.extend( palette[:NUM_TOP_COLORS] )
    hsv_colors.extend( [ np.array(rgb_to_hsv(*list(c/255))) for c in palette[:NUM_TOP_COLORS] ] )

plt.hist(stds, bins=10)
plt.savefig('stddev.png')

def make_canvas(array, pixel_size=10, name='canvas'):
    side = math.ceil(math.sqrt(len(array)))
    side_pixel = side*pixel_size
    canvas = np.zeros((side_pixel, side_pixel,3))
    print(canvas.shape)

    for idx, rgb_color in enumerate(array): 
        i = idx%side
        j = idx//side
        canvas[j*pixel_size:(j+1)*pixel_size+pixel_size, i*pixel_size:(i+1)*pixel_size] = rgb_color

    cv2.imwrite(f'{name}.jpg', canvas)

make_canvas(rgb_colors, name='canvas')

filter_color = np.array(stds) > VAR_THRESH
rgb_colors = np.array(rgb_colors)
rgb_colors_filtered = rgb_colors[filter_color]
make_canvas(rgb_colors_filtered, name='canvas_filtered')
print(f'After STD filtering, num colors from {len(rgb_colors)} to {len(rgb_colors_filtered)}')

hsv_colors = np.array(hsv_colors)
hsv_colors_filtered = hsv_colors[filter_color]

kmeans = KMeans(n_clusters=K, random_state=SEED)
# kmeans.fit(rgb_colors_filtered)
kmeans.fit(hsv_colors_filtered)
color_centers = kmeans.cluster_centers_
clus_idxes = list(kmeans.labels_)

print(color_centers)

percent=[]
for i in range(K):
  j=clus_idxes.count(i)
  j=j/(len(clus_idxes))
  percent.append(j)
print(percent)


plt.clf() 
rgb_color_centers = np.array([ rgb_to_hsv(*list(c)) for c in color_centers ])
plt.pie(percent,colors=np.array(rgb_color_centers),labels=np.arange(K))
# plt.pie(percent,colors=np.array(color_centers/255),labels=np.arange(K))
plt.savefig('pie.png')


# from colorthief import MMCQ 

# cmap = MMCQ.quantize(rgb_colors, K)
# palette = cmap.palette

# print(palette)

import ipdb; ipdb.set_trace()