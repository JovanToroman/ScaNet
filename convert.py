import os
from shutil import copyfile
import sqlite3 as sql

from PIL import Image
import imageio as io
from sklearn.feature_extraction import image
import skimage
import cv2

from image_align import align_image
from scan import detect_paper
from db import insert_into_db, check_if_image_exists

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

def extract_patches(root_path_scanned, scanned_folders, path_originals, patch_dims, patches_per_image=40):
    try:
        count = 0
        for folder in scanned_folders:
            for index, f in enumerate(sorted(os.listdir(os.path.join(root_path_scanned, folder)))):
                original_image_name = f.split('_')[0] + '.jpg'
                one_image = io.imread(os.path.join(os.path.join(root_path_scanned, folder), f))
                two_image = io.imread(os.path.join(path_originals, original_image_name))
                if two_image.shape != one_image.shape:
                    two_image = cv2.resize(two_image, (one_image.shape[1], one_image.shape[0])).astype('uint8')
                patches = image.extract_patches_2d(one_image, patch_dims, patches_per_image, 100)
                patches2 = image.extract_patches_2d(two_image, patch_dims, patches_per_image, 100)
                for index2, patch in enumerate(patches):
                    if not (check_if_image_is_white(patch) or check_if_image_is_white(patches2[index2])):
                        io.imsave(os.path.join('dped/test/training_data/test', str(count) + '.jpg'), patch)
                        io.imsave(os.path.join('dped/test/training_data/original', str(count) + '.jpg'), patches2[index2])
                        count += 1
    except Exception as e:
        print(e)
        return


def align_images(path_originals, path_scanned, output_path, phone, conn):
    scanned = [s for s in os.listdir(path_scanned)]
    for image in scanned:
        if check_if_image_exists(image, 2, phone, conn):
            print("Image {} for phase {} and phone {} already exists in db".format(image, 2, phone))
            continue
        original_image_name = image.split('_')[0] + '.jpg'
        result = align_image(os.path.join(path_originals, original_image_name), os.path.join(path_scanned, image),
                    os.path.join(output_path, image))
        if result:
            status, ssim = result
            insert_into_db(image, status, 2, ssim, None, phone, conn)


def extract_sheets_of_paper(path_originals, path_transformed):
    [detect_paper(os.path.join(path_originals, o), os.path.join(path_transformed, o))
     for o in os.listdir(path_originals) if not o.startswith('.')]

def gaussian_blur(path_to_clear_images):
    """Edits images in-place to add gaussian noise to them"""
    try:
        count = 0
        for index, f in enumerate(sorted(os.listdir(path_to_clear_images))):
            clear_image = io.imread(os.path.join(path_to_clear_images, f))
            blurred_image = skimage.util.random_noise(clear_image, mode='gaussian')
            io.imsave(os.path.join(path_to_clear_images, f), blurred_image)
    except Exception as _e:
        return


def separate_test_data(input_path, output_path_test, test_ratio=0.01):
    count = 0
    for file in os.listdir(input_path):
        if count % (1/test_ratio) == 0:
            os.rename(os.path.join(input_path, file), os.path.join(output_path_test, file))
        count += 1



phones = ['HTC', 'LG_G3', 'Motorola', 'Samsung_A3', 'Samsung_S20', 'Sony_Xperia']


## code bit for phase 2 (ORB alignment)
# conn = sql.connect('../image.sqlite')
# for phone in phones:
#     align_images('origim',
#                  os.path.join('../captured_images/transposed', phone),
#                  os.path.join('../captured_images/aligned_orb', phone),
#                  phone,
#                  conn)

## separate test data from dataset
# for phone in phones:
#     separate_test_data(os.path.join('../captured_images/aligned', phone),
#                        os.path.join('../captured_images/aligned/test', phone))


extract_patches('../captured_images/aligned', phones, 'origim',
                (100,100), 40)

# rename(r'C:\Users\Jovan\Documents\Master_rad\zajem\originals')
# extract_sheets_of_paper(r'C:\Users\Jovan\Documents\Master_rad\zajem\samsung_tatin\high_lighting',
#              r'C:\Users\Jovan\Documents\Master_rad\zajem\samsung_tatin\high_lighting_processed')
# resize(r'C:\Users\Jovan\Documents\Master_rad\zajem\originals - Copy', [1080, 1528])
# gaussian_blur('/opt/workspace/host_storage_hdd/projects/dped_copy/dped/test/test_data/full_size_test_images')