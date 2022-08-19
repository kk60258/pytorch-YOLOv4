import cv2
import os
import glob
import pathlib
from tqdm import tqdm
source = '/home/jason/Downloads/val2017'
out = '/home/jason/Downloads/val2017_512'
images = glob.glob(os.path.join(source, '*.jpg'))

pathlib.Path(out).mkdir(parents=True, exist_ok=True)
for path in tqdm(images):
    basename = os.path.basename(path)
    img = cv2.imread(path)
    img = cv2.resize(img, (512, 512))
    cv2.imwrite(os.path.join(out, basename), img)