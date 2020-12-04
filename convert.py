import os
from shutil import copyfile
import sqlite3 as sql
import pathlib

from PIL import Image, ImageEnhance
import imageio as io
from sklearn.feature_extraction import image
import skimage
import cv2
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from matplotlib import pyplot as plt

from image_align import align_image
from scan import detect_paper
from db import insert_into_db, check_if_image_exists
from utils import concat_images, bring_to_same_resolution

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
                try:
                    patches = image.extract_patches_2d(one_image, patch_dims, patches_per_image, 100)
                    patches2 = image.extract_patches_2d(two_image, patch_dims, patches_per_image, 100)
                except:
                    continue
                for index2, patch in enumerate(patches):
                    ## do not discard all white or black images because we don't want our data to be biased
                    # if not (check_if_image_is_white(patch) or check_if_image_is_white(patches2[index2])):
                    io.imsave(os.path.join('dped/test/training_data/test2', str(count) + '.jpg'), patch)
                    io.imsave(os.path.join('dped/test/training_data/original2', str(count) + '.jpg'), patches2[index2])
                    count += 1
                    if count % 1000 == 0:
                        print(count)
    except Exception as e:
        print(e)
        # return

## extract the same
# def extract_patches(root_path_scanned, scanned_folders, path_originals, patch_dims, patches_per_image=40):
#     try:
#         count = 579900
#         ## for folder in scanned_folders: # when using phone directories
#         for index, f in enumerate(sorted(os.listdir(root_path_scanned))):#os.path.join(root_path_scanned, folder)))):
#             original_image_name = f#f.split('_')[0] + '.jpg'
#             one_image = io.imread(os.path.join(root_path_scanned, f))#os.path.join(root_path_scanned, folder), f))
#             two_image = io.imread(os.path.join(path_originals, original_image_name))
#             if two_image.shape != one_image.shape:
#                 two_image = cv2.resize(two_image, (one_image.shape[1], one_image.shape[0])).astype('uint8')
#             patches = image.extract_patches_2d(one_image, patch_dims, patches_per_image, 100)
#             patches2 = image.extract_patches_2d(two_image, patch_dims, patches_per_image, 100)
#             for index2, patch in enumerate(patches):
#                 ## do not discard all white or black images because we don't want our data to be biased
#                 # if not (check_if_image_is_white(patch) or check_if_image_is_white(patches2[index2])):
#                 io.imsave(os.path.join('dped/test/training_data/test_test', str(count) + '.jpg'), patch)
#                 io.imsave(os.path.join('dped/test/training_data/original_test', str(count) + '.jpg'), patches2[index2])
#                 count += 1
#                 if count % 1000 == 0:
#                     print(count)
#     except Exception as e:
#         print(e)
#         return


def align_images(path_originals, path_scanned, output_path, phone=None, conn=None):
    scanned = [s for s in os.listdir(path_scanned)]
    total_mse = 0
    total_ssim = 0
    count = 0
    for image in scanned:
        if conn and check_if_image_exists(image, 2, phone, conn):
            print("Image {} for phase {} and phone {} already exists in db".format(image, 2, phone))
            continue
        original_image_name = image.split('.')[0] + '.png'
        result = align_image(os.path.join(path_originals, original_image_name), os.path.join(path_scanned, image),
                    os.path.join(output_path, image))
        try:
            status, ssim, mse = result
            total_mse += mse
            total_ssim += ssim
            count += 1
        except:
            pass
        if conn and result:
            status, ssim = result
            insert_into_db(image, status, 2, ssim, None, phone, conn)
    print('Average ssim: {}; average mse: {}'.format(total_ssim/count, total_mse/count))


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

def concat_items(images_path, capture_folder, manual_contrast_folder, our_method_folder, groundtruth_folder, output_folder):
    for im in os.listdir(os.path.join(images_path, capture_folder)):
        captured_image = Image.open(os.path.join(images_path, capture_folder, im))
        manual_contrast_image = Image.open(os.path.join(images_path, manual_contrast_folder, im))
        our_method_image = Image.open(os.path.join(images_path, our_method_folder, im))
        groundtruth_image = Image.open(os.path.join(images_path, groundtruth_folder, im))
        concat_images([captured_image, manual_contrast_image, our_method_image, groundtruth_image],
                      os.path.join(images_path, output_folder, im))


def change_contrast(im, factor):
    im = Image.fromarray(im)
    enhancer = ImageEnhance.Contrast(im)
    return np.array(enhancer.enhance(factor))


def calc_mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def find_best_contrast_level_wrt_MSE(src, dst, output_dir = None, image_name = None):
    """Tries different contrast levels on src image to make it as similar to dst w.r.t. MSE."""
    src, dst = bring_to_same_resolution(src, dst)
    contrast_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                       2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    best_ssim = 0
    best_mse = float('inf')
    res = src
    best_lvl = 1
    print('ssim beginning: {}'.format(tf.image.ssim(tf.convert_to_tensor(src), tf.convert_to_tensor(dst), 1.0)))
    for lvl in contrast_levels:
        print('now at lvl {}'.format(lvl))
        updated_image = change_contrast(src, lvl)
        ssim = float(tf.image.ssim(tf.convert_to_tensor(updated_image), tf.convert_to_tensor(dst), 1.0))
        mse = tf.losses.mse()#calc_mse(updated_image, dst)
        if mse < best_mse:
            best_ssim = ssim
            best_mse = mse
            res = updated_image
            best_lvl = lvl
    print('best ssim: {}, best lvl: {}'.format(best_ssim, best_lvl))
    if output_dir and image_name:
        io.imsave(os.path.join(output_dir,
                               '{}_{}_{}_{}{}'.format(image_name.split('.')[0], best_lvl, best_mse, best_ssim,
                                                      '.jpeg')), res)
    return (best_mse, best_ssim)


def find_best_contrast_level_wrt_SSIM(src, dst, output_dir = None, image_name = None):
    """Tries different contrast levels on src image to make it as similar to dst w.r.t. SSIM."""
    src, dst = bring_to_same_resolution(src, dst)
    contrast_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                       2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    best_ssim = 0
    best_mse = float('inf')
    res = src
    best_lvl = 1
    print('ssim beginning: {}'.format(tf.image.ssim(tf.convert_to_tensor(src), tf.convert_to_tensor(dst), 1.0)))
    for lvl in contrast_levels:
        print('now at lvl {}'.format(lvl))
        updated_image = change_contrast(src, lvl)
        ssim = float(tf.image.ssim(tf.convert_to_tensor(updated_image), tf.convert_to_tensor(dst), 1.0))
        mse = calc_mse(updated_image, dst)
        if ssim > best_ssim:
            best_ssim = ssim
            best_mse = mse
            res = updated_image
            best_lvl = lvl
    print('best ssim: {}, best lvl: {}'.format(best_ssim, best_lvl))
    if output_dir and image_name:
        io.imsave(os.path.join(output_dir,
                               '{}_{}_{}_{}{}'.format(image_name.split('.')[0], best_lvl, best_mse, best_ssim,
                                                      '.jpeg')), res)
    return (best_mse, best_ssim)


def find_best_contrast_level_wrt_SSIM_and_MSE(src, dst, output_dir = None, image_name = None):
    """Tries different contrast levels on src image to make it as similar to dst w.r.t. SSIM."""
    src, dst = bring_to_same_resolution(src, dst)
    contrast_levels = [0.1, 0.2,0.3, 0.4, 0.5,0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,2.0, 3.0, 4.0, 5.0,6.0,7.0,8.0,9.0,10.0]
    best_ssim = 0
    best_mse = float('inf')
    res = src
    best_lvl = 1
    print('ssim beginning: {}'.format(tf.image.ssim(tf.convert_to_tensor(src), tf.convert_to_tensor(dst), 1.0)))
    for lvl in contrast_levels:
        print('now at lvl {}'.format(lvl))
        updated_image = change_contrast(src, lvl)
        ssim = float(tf.image.ssim(tf.convert_to_tensor(updated_image), tf.convert_to_tensor(dst), 1.0))
        mse = calc_mse(updated_image, dst)
        if ssim > best_ssim and mse < best_mse:
            best_ssim = ssim
            best_mse = mse
            res = updated_image
            best_lvl = lvl
    print('best ssim: {}, best lvl: {}'.format(best_ssim, best_lvl))
    if output_dir and image_name:
        io.imsave(os.path.join(output_dir, '{}_{}_{}_{}{}'.format(image_name.split('.')[0],best_lvl, best_mse, best_ssim, '.jpeg')), res)
    return (best_mse, best_ssim)


def plot_dropouts(points_no_dropout, points_50, points_90):
    x = [5,10,15,20,25,30,35,40,45,50]
    plt.plot(x, points_no_dropout, label="No dropout")
    plt.plot(x, points_50, label="50% dropout")
    plt.plot(x, points_90, label="90% dropout")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()



phones = ['HTC', 'LG_G3', 'Motorola', 'Samsung_A3', 'Samsung_S20', 'Sony_Xperia']


########################################################
#              EXAMPLES USAGES                         #
#                                                      #
########################################################


# plot_dropouts([0.173, 0.130, 0.112, 0.099, 0.091, 0.083, 0.080, 0.074, 0.073, 0.070],
#               [0.217, 0.159, 0.137, 0.125, 0.118, 0.113, 0.110, 0.107, 0.104, 0.102],
#               [0.300, 0.184, 0.152, 0.136, 0.127, 0.118, 0.111, 0.107, 0.105, 0.103])

## simple contrast enhancer
# total_mse = 0
# total_ssim = 0
# count = 0
# for image in os.listdir('disjoint/captured_new'):
#     original_image_name = image.split('.')[0] + '.png'#image.split('_')[0] + '.png'
#     ssim, mse = find_best_contrast_level_wrt_SSIM_and_MSE(io.imread(os.path.join('disjoint/captured_new', image)), #'dped/test/test_data/full_size_test_images/test_images_disjoint/4_0.jpeg'),
#                io.imread(os.path.join('disjoint/originals_new', original_image_name)), 'output_disjoint_enhanced_contrast_new', image)
#     total_mse += mse
#     total_ssim += ssim
#     count += 1
# print('Average ssim: {}; average mse: {}'.format(total_ssim/count, total_mse/count))
#
# align_images('disjoint/originals_new', 'disjoint/captured_new', 'output_disjoint_enhanced_aligned_new')

## concat results for paper unet
# input = Image.open('disjoint/captured_new/54.jpg')
# adobe = Image.open('output_disjoint_enhanced_adobe_new/54_captured_adobe.jpg')
# contrast = Image.open('output_disjoint_enhanced_contrast_new/54_0.9_13072.588539642584_0.002895546844229102.jpeg')
# ours = Image.open('disjoint/enhanced_new/54.jpg')
# original = Image.open('disjoint/originals_new/54.png')
# concat_images([
#     input, adobe,
#     # contrast,
#     ours,
#                original
#                ], 'output_comparison_contrast_and_ours_unet_new/54_adobe.jpeg')

## concat results for paper resnet
# input = Image.open('dped/test/test_data/full_size_test_images/test_images_disjoint_aligned/1_1.jpeg')
# adobe = Image.open('output_disjoint_enhanced_adobe/1_1_13824.836380998267_0.07230796664953232_.jpeg')
# # contrast = Image.open('output_disjoint_enhanced_contrast/5_1_0.5_22124.474580528728_0.08854147791862488.jpeg')
# ours = Image.open('output_disjoint_enhanced_aligned_resnet/1_1_7799.378717669404_0.12113260477781296_.jpeg')
# original = Image.open('disjoint/originals/1.png')
# concat_images([
#     input,
#     adobe,
#     # contrast,
#     ours,
#                original
#                ], 'output_comparison_contrast_and_ours_resnet/1_1_adobe.jpeg')


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


# extract_patches('../captured_images/aligned', phones, 'origim',
#                 (512,512), 5)

# rename('dped/test/test_data/full_size_test_images')
# extract_sheets_of_paper(r'C:\Users\Jovan\Documents\Master_rad\zajem\samsung_tatin\high_lighting',
#              r'C:\Users\Jovan\Documents\Master_rad\zajem\samsung_tatin\high_lighting_processed')
# resize(r'C:\Users\Jovan\Documents\Master_rad\zajem\originals - Copy', [1080, 1528])
# gaussian_blur('/opt/workspace/host_storage_hdd/projects/dped_copy/dped/test/test_data/full_size_test_images')
