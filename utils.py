import numpy as np
import PIL
from PIL import Image
import cv2

### Resnet
import scipy.stats as st
import tensorflow as tf
import sys

from functools import reduce


def calc_mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def bring_to_same_resolution(im1, im2):
    # im1 = cv2.rotate(im1, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    height1, width1, _ = im1.shape
    height2, width2, _ = im2.shape
    min_height = min(height1, height2)
    min_width = min(width1, width2)
    return (cv2.resize(im1, (min_width, min_height)), \
           cv2.resize(im2, (min_width, min_height)))

def concat_images(imgs, output_file_name):
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))

    # save that beautiful picture
    imgs_comb = PIL.Image.fromarray(imgs_comb)
    imgs_comb.save(output_file_name)
    

def resize_image_if_uneven_width_or_height(image):
    shape = image.shape
    width = shape[0] if shape[0] % 2 == 0 else shape[0] - 1
    while width % 16 != 0:
        width -= 1
    height = shape[1] if shape[1] % 2 == 0 else shape[1] - 1
    while height % 16 != 0:
        height -= 1
    return cv2.rotate(cv2.resize(image, (height, width)), rotateCode=cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

### used for Resnet

def log10(x):
  numerator = tf.compat.v1.log(x)
  denominator = tf.compat.v1.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d for d in tensor.get_shape()[1:]), 1)

def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter
