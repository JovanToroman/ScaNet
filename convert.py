import os
from shutil import copyfile

from PIL import Image
from scipy import misc
from sklearn.feature_extraction import image

from image_align import align_image

path=r'C:\Users\Jovan\Documents\Master_rad\testni_zajem\originals - Copy'
path2=r'C:\Users\Jovan\Documents\Master_rad\testni_zajem\sams_a3_cropped - Copy'
path3=r'C:\Users\Jovan\Documents\Master_rad\testni_zajem\aligned'


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

def align_images(path_originals, path_scanned, output_path):
    originals = [o for o in os.listdir(path_originals)]
    scanned = [s for s in os.listdir(path_scanned)]
    for i in range(min(len(originals), len(scanned))):
        align_image(os.path.join(path_originals, originals[i]), os.path.join(path_scanned, scanned[i]),
                    os.path.join(output_path, str(i) + ".jpg"))


extract_patches(r'C:\Users\Jovan\Documents\Master_rad\testni_zajem\originals - Copy', [100,100])
# align_images(path, path2, path3)