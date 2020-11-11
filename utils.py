from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# ----------help function with data io------------------- #

def get_image(image_path, image_size, is_crop = True, resize_w = 64, mode = None):
    return transform(imread(image_path, mode), image_size, is_crop, resize_w)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def save_image(image, image_path):
    image = 255 * inverse_transform(image)
    image = np.clip(image, 0, 255).astype(np.uint8)
    if len(image.shape) == 3 and image.shape[-1] == 1:
        image = np.reshape(image, (image.shape[0], image.shape[1]))
    scipy.misc.imsave(image_path, image)


def imread(path, mode = None):
    if mode == 'RGB':
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
    else:
        img = scipy.misc.imread(path).astype(np.float)
        if len(img.shape) < 3:
            img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        return img


def denormalize(images):
    img = 255 * inverse_transform(images)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def denormalize_with_merge(images, size):
    merge_img =  255 * merge(inverse_transform(images), size)
    merge_img = np.clip(merge_img, 0, 255).astype(np.uint8)
    return merge_img


def merge(images, size):
    if(len(images.shape) > 3):
        h, w, c = images.shape[1], images.shape[2], images.shape[3]
        img = np.zeros((int(h * size[0]), int(w * size[1]), c))

        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image

        if c == 1:
            img = img.reshape(img.shape[0], img.shape[1])

    else:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((int(h * size[0]), int(w * size[1])))

        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image

    return img


def imsave(images, size, path):
    merge_img = 255 * merge(images, size)
    merge_img = np.clip(merge_img, 0, 255).astype(np.uint8)
    return scipy.misc.imsave(path, merge_img)


def center_crop(x, crop_h, crop_w = None, resize_w = 64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_w, resize_w])


def transform(image, npx = 64, is_crop = True, resize_w = 64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w = resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.