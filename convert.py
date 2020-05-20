import os
from shutil import copyfile

from PIL import Image
import imageio as io
from sklearn.feature_extraction import image

from image_align import align_image
from scan import detect_paper

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
        [os.rename(os.path.join(path, f), os.path.join(path, str(int(f.split('.')[0]) -20486) + '.jpg')) for index, f in
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
    black_thresh = 20
    n_black = 0
    for pixel in pixels:
        if pixel > white_thresh:
            n_white += 1
        if pixel < black_thresh:
            n_black += 1
    n = len(pixels)

    return (n_white / float(n)) > 0.95 or (n_black / float(n)) > 0.95

def extract_patches(path_scanned, path_originals, patch_dims):
    try:
        count = 0
        for index, f in enumerate(sorted(os.listdir(path_scanned))):
            one_image = io.imread(os.path.join(path_scanned, f))
            two_image = io.imread(os.path.join(path_originals, f))
            patches = image.extract_patches_2d(one_image, patch_dims, 200, 100)
            patches2 = image.extract_patches_2d(two_image, patch_dims, 200, 100)
            for index2, patch in enumerate(patches):
                # if not (check_if_image_is_white(patch) or check_if_image_is_white(patches2[index2])):
                io.imsave(os.path.join(r'C:\Users\Jovan\Documents\Master_rad\DPED\dped\test\training_data\test', str(count) + '.jpg'), patch)
                io.imsave(os.path.join(r'C:\Users\Jovan\Documents\Master_rad\DPED\dped\test\training_data\canon', str(count) + '.jpg'), patches2[index2])
                count += 1
    except Exception as e:
        print(e)
        return

def align_images(path_originals, path_scanned, output_path):
    originals = [o for o in os.listdir(path_originals)]
    scanned = [s for s in os.listdir(path_scanned)]
    for i in range(min(len(originals), len(scanned))):
        align_image(os.path.join(path_originals, originals[i]), os.path.join(path_scanned, scanned[i]),
                    os.path.join(output_path, originals[i].split('.')[0] + ".jpg"))

def extract_sheets_of_paper(path_originals, path_transformed):
    [detect_paper(os.path.join(path_originals, o), os.path.join(path_transformed, o))
     for o in os.listdir(path_originals) if not o.startswith('.')]



extract_patches(r'C:\Users\Jovan\Documents\Master_rad\zajem\samsung_s10e\low_lighting_aligned_good',
                r'C:\Users\Jovan\Documents\Master_rad\zajem\originals2 - Copy - Copy',
                [100,100])
# align_images(r'C:\Users\Jovan\Documents\Master_rad\zajem\originals2 - Copy',
#              r'C:\Users\Jovan\Documents\Master_rad\zajem\samsung_tatin\low_lighting_processed',
#              r'C:\Users\Jovan\Documents\Master_rad\zajem\samsung_tatin\low_lighting_aligned')
# rename(r'C:\Users\Jovan\Documents\Master_rad\zajem\originals2 - Copy - Copy\patches')
# extract_sheets_of_paper(r'C:\Users\Jovan\Documents\Master_rad\zajem\samsung_tatin\high_lighting',
#              r'C:\Users\Jovan\Documents\Master_rad\zajem\samsung_tatin\high_lighting_processed')
# resize(r'C:\Users\Jovan\Documents\Master_rad\DPED\dped\test\test_data\patches\test', [1080, 1528])