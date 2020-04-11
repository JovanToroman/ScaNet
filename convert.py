import os
from shutil import copyfile

import numpy as np
from PIL import Image
from scipy import misc
import tensorflow.compat.v1 as tf
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image

path=r'C:\Users\Jovan\Downloads\Manuscript_1587_Panagiotopoulou.jpg'
path2=r'C:\Users\Jovan\Documents\Master_rad\testni_zajem\sams_a3 - Copy'
path3=r'C:\Users\Jovan\Documents\Master_rad\DPED\dped\test\training_data\canon'
path4=r'C:\Users\Jovan\Documents\Master_rad\DPED\dped\test\training_data\test'


def rename_multiple(path, path_to):
    try:
        [[copyfile(os.path.join(path, f), os.path.join(path_to, str(i) + '.jpg')) for i in range(96)]
         for index, f in enumerate(os.listdir(path)) if not f.startswith('.')]
    except Exception as e:
        print(e)

def rename(path):
    try:
        [os.rename(os.path.join(path, f), os.path.join(path, str(index) + '.jpg')) for index, f in
         enumerate(os.listdir(path)) if not f.startswith('.')]
    except Exception as e:
        print(e)


def resize(path, dimensions):
    for index, f in enumerate(os.listdir(path)):
        foo = Image.open(os.path.join(path, f))
         # I downsize the image with an ANTIALIAS filter (gives the highest quality)
        foo = foo.resize(dimensions,Image.ANTIALIAS)
        foo.save(os.path.join(path, f),quality=95)
         # The saved downsized image size is 24.8kb
        # foo.save("path\\to\\save\\image_scaled_opt.jpg",optimize=True,quality=95)


def extract_patches(path, patch_dims):
    width, height = patch_dims
    for index, f in enumerate(os.listdir(path)):
        one_image = misc.imread(os.path.join(path, f))
        patches = image.extract_patches_2d(one_image, (width, height), max_patches=100, random_state=100)
        for index2, patch in enumerate(patches):
            misc.imsave(os.path.join(path, "patches",  str(index * len(patches) + index2) + ".jpg"), patch)

extract_patches(r'C:\Users\Jovan\Documents\Master_rad\testni_zajem\sams_a3_cropped - Copy', [100,100])