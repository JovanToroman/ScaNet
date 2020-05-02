import os
from shutil import copyfile

from PIL import Image
from scipy import misc
from sklearn.feature_extraction import image

from image_align import align_image

path=r'C:\Users\Jovan\Documents\Master_rad\testni_zajem\a'
path2=r'C:\Users\Jovan\Documents\Master_rad\testni_zajem\b'
path3=r'C:\Users\Jovan\Documents\Master_rad\testni_zajem\aligned'


def rename_multiple(path, path_to):
    try:
        [[copyfile(os.path.join(path, f), os.path.join(path_to, str(i) + '.jpg')) for i in range(96)]
         for index, f in enumerate(os.listdir(path)) if not f.startswith('.')]
    except Exception as e:
        print(e)


def duplicate_originals(path, path_to, number_of_perspectives):
    """Duplicate them to match scanned images of the same original file in different perspectives"""
    try:
        [[copyfile(os.path.join(path, f), os.path.join(path_to, str(int(f.split('.')[0])*number_of_perspectives + i) + '.jpg'))
          for i in range(0, number_of_perspectives)]
         for index, f in enumerate(os.listdir(path))] #if not f.startswith('.')] add this if problems
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

def check_if_image_is_white(image):
    img = Image.fromarray(image).convert('L')
    pixels = img.getdata()  # get the pixels as a flattened sequence
    white_thresh = 130
    n_white = 0
    for pixel in pixels:
        if pixel > white_thresh:
            n_white += 1
    n = len(pixels)

    return (n_white / float(n)) > 0.95

def extract_patches(path_scanned, path_originals, patch_dims):
    try:
        count = 0
        for index, f in enumerate(sorted(os.listdir(path_scanned))):
            one_image = misc.imread(os.path.join(path_scanned, f))
            two_image = misc.imread(os.path.join(path_originals, f))
            patches = image.extract_patches_2d(one_image, patch_dims, 100, 100)
            patches2 = image.extract_patches_2d(two_image, patch_dims, 100, 100)
            for index2, patch in enumerate(patches):
                if not (check_if_image_is_white(patch) or check_if_image_is_white(patches2[index2])):
                    misc.imsave(os.path.join(path_scanned, "patches1", str(count) + '.jpg'), patch)
                    misc.imsave(os.path.join(path_originals, "patches1", str(count) + '.jpg'), patches2[index2])
                    count += 1
    except Exception as e:
        return

def align_images(path_originals, path_scanned, output_path):
    originals = [o for o in os.listdir(path_originals)]
    scanned = [s for s in os.listdir(path_scanned)]
    for i in range(min(len(originals), len(scanned))):
        align_image(os.path.join(path_originals, originals[i]), os.path.join(path_scanned, scanned[i]),
                    os.path.join(output_path, originals[i].split('.')[0] + ".jpg"))


# extract_patches(r'C:\Users\Jovan\Documents\Master_rad\testni_zajem\originals - Copy', [100,100])
# extract_patches(r'C:\Users\Jovan\Documents\Master_rad\testni_zajem\aligned_good',
#                 r'C:\Users\Jovan\Documents\Master_rad\testni_zajem\originals - Copy', [100,100])
# align_images(path, path2, path3)
# duplicate_originals(r'C:\Users\Jovan\Documents\Master_rad\testni_zajem\originals', r'C:\Users\Jovan\Documents\Master_rad\testni_zajem\originals_mock', 8)
# rename(r'C:\Users\Jovan\Documents\Master_rad\testni_zajem\aligned_good_high_res')
extract_patches(r'C:\Users\Jovan\Documents\Master_rad\testni_zajem\aligned_good_high_res',
                r'C:\Users\Jovan\Documents\Master_rad\testni_zajem\originals_mock', [100,100])