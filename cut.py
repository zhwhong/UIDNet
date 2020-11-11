import os
import scipy.misc
import itertools
from glob import glob
import numpy as np


SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]
PATIENT_LIST = ['L067', 'L096', 'L109', 'L143', 'L192', 'L286', 'L291', 'L310', 'L333', 'L506']


def dataset_files(root):
    """Returns a list of all image files in the given directory"""
    return list(itertools.chain.from_iterable(
        glob(os.path.join(root, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS))

# The HU bound to filter lung
# def normalize(image, MIN_BOUND = -1000.0, MAX_BOUND = 400.0):

# The HU bound to filter noise part
def normalize(image, MIN_BOUND = -160.0, MAX_BOUND = 240.0):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.0
    image[image < 0] = 0.0
    return image


def cut(src, dst):
    # suffix_clean = '_full_3mm.npz'
    # suffix_noise = '_quarter_3mm.npz'
    suffix_clean = '_full_1mm.npz'
    suffix_noise = '_quarter_1mm.npz'
    dst_clean = os.path.join(dst, 'cut_clean')
    dst_noise = os.path.join(dst, 'cut_noise')
    if not os.path.exists(dst_clean):
        os.makedirs(dst_clean)
    if not os.path.exists(dst_noise):
        os.makedirs(dst_noise)

    i = 0
    for id in PATIENT_LIST:
        #if i > 0:
        #    break
        i += 1

        print('processing patient id : %s' % (id,))
        npz_clean = id + suffix_clean
        npz_noise = id + suffix_noise

        cube_clean = np.load(os.path.join(src, npz_clean))['arr_0']
        # cube_clean = np.transpose(cube_clean, (2, 1, 0))
        cube_clean = normalize(cube_clean)   # from 0 to 1

        cube_noise = np.load(os.path.join(src, npz_noise))['arr_0']
        # cube_noise = np.transpose(cube_noise, (2, 1, 0))
        cube_noise = normalize(cube_noise)  # from 0 to 1

        channel, width, height = cube_clean.shape
        # for j in range(0, channel, 3):
        for j in range(channel):
            print('[Patient %d : %s] Processing CT slice %d/%d ...' % (i, id, j + 1, channel))
            slice_clean = cube_clean[j]
            slice_noise = cube_noise[j]
            w = 0
            while w <= width - 64:
                h = 0
                while h <= height - 64:
                    patch_clean = slice_clean[w:w + 64, h:h + 64]
                    patch_noise = slice_noise[w:w + 64, h:h + 64]
                    # if patch_clean.mean() < 0.2:
                    if patch_clean.mean() < 0.25 or patch_clean.mean() > 0.85:
                        h += 32
                        continue
                    scipy.misc.imsave(os.path.join(dst_clean, '%s_%04d_%03d_%03d.bmp' % (id, j + 1, w, h)),
                                      (255 * patch_clean).astype(np.uint8))
                    scipy.misc.imsave(os.path.join(dst_noise, '%s_%04d_%03d_%03d.bmp' % (id, j + 1, w, h)),
                                      (255 * patch_noise).astype(np.uint8))
                    h += 32
                w += 32


def cut_single(src, dst, mode = 'noise'):
    suffix = '_full_3mm.npz' if mode == 'clean' else '_quarter_3mm.npz'
    dst = os.path.join(dst, 'cut_'+ mode)
    if not os.path.exists(dst):
        os.makedirs(dst)

    i = 0
    for id in PATIENT_LIST:
        if i > 0:
            break
        i += 1

        print('processing patient id : %s' % (id,))
        npz_file = id + suffix

        cube = np.load(os.path.join(src, npz_file))['arr_0']
        cube = np.transpose(cube, (2, 1, 0))
        cube = normalize(cube)   # from 0 to 1

        channel, width, height = cube.shape
        for j in range(60, 70):
            print('[Patient %d : %s] Processing CT slice %d/%d ...' % (i, id, j + 1, channel))
            slice = cube[j]
            w = 0
            while w <= width - 64:
                h = 0
                while h <= height - 64:
                    patch = slice[w:w + 64, h:h + 64]
                    if patch.mean() < 0.2:
                        h += 32
                        continue
                    scipy.misc.imsave(os.path.join(dst, '%s_%03d_%d_%d.png' % (id, j + 1, w, h)),
                                      (255*patch).astype(np.uint8))
                    h += 32
                w += 32


def view(src, dst):
    i = 0
    for id in PATIENT_LIST:
        if i > 1:
            break
        i += 1

        print('processing patient id : %s' % (id,))
        subdst = os.path.join(dst, id)
        if not os.path.exists(subdst):
            os.makedirs(subdst)

        # deal with full dose CT image
        npz_clean = id + '_full_3mm.npz'
        cube = np.load(os.path.join(src, npz_clean))['arr_0']
        cube = np.transpose(cube, (2, 1, 0))
        cube = normalize(cube)   # from 0 to 1

        channel, width, height = cube.shape
        for j in range(channel):
            print('[Patient %d : %s] Processing CT slice %d/%d (clean) ...' % (i, id, j + 1, channel))
            slice = cube[j]
            scipy.misc.imsave(os.path.join(subdst, 'clean_%03d.png' % (j + 1)), (255 * slice).astype(np.uint8))

        # deal with low dose CT image
        npz_noise = id + '_quarter_3mm.npz'
        cube = np.load(os.path.join(src, npz_noise))['arr_0']
        cube = np.transpose(cube, (2, 1, 0))
        cube = normalize(cube)  # from 0 to 1

        channel, width, height = cube.shape
        for j in range(channel):
            print('[Patient %d : %s] Processing CT slice %d/%d (noise) ...' % (i, id, j + 1, channel))
            slice = cube[j]
            scipy.misc.imsave(os.path.join(subdst, 'noise_%03d.png' % (j + 1)), (255 * slice).astype(np.uint8))




def cut_test(src, dst):
    suffix_clean = '_full_3mm.npz'
    suffix_noise = '_quarter_3mm.npz'
    # suffix_clean = '_full_1mm.npz'
    # suffix_noise = '_quarter_1mm.npz'
    dst_clean = os.path.join(dst, 'cut_clean')
    dst_noise = os.path.join(dst, 'cut_noise')
    if not os.path.exists(dst_clean):
        os.makedirs(dst_clean)
    if not os.path.exists(dst_noise):
        os.makedirs(dst_noise)

    i = 0
    cnt = 0
    for id in PATIENT_LIST:
        #if i > 0:
        #    break
        i += 1

        print('processing patient id : %s' % (id,))
        npz_clean = id + suffix_clean
        npz_noise = id + suffix_noise

        cube_clean = np.load(os.path.join(src, npz_clean))['arr_0']
        # cube_clean = np.transpose(cube_clean, (2, 1, 0))
        cube_clean = normalize(cube_clean)   # from 0 to 1

        cube_noise = np.load(os.path.join(src, npz_noise))['arr_0']
        # cube_noise = np.transpose(cube_noise, (2, 1, 0))
        cube_noise = normalize(cube_noise)  # from 0 to 1

        channel, width, height = cube_clean.shape
        # for j in range(0, channel, 3):
        for j in range(channel):
            print('[Patient %d : %s] Processing CT slice %d/%d ...' % (i, id, j + 1, channel))
            slice_clean = cube_clean[j]
            slice_noise = cube_noise[j]
            w = 0
            while w <= width - 64:
                h = 0
                while h <= height - 64:
                    patch_clean = slice_clean[w:w + 64, h:h + 64]
                    patch_noise = slice_noise[w:w + 64, h:h + 64]
                    # if patch_clean.mean() < 0.2:
                    if patch_clean.mean() < 0.25 or patch_clean.mean() > 0.85:
                        h += 32
                        continue
                    scipy.misc.imsave(os.path.join(dst_clean, '%s_%04d_%03d_%03d.bmp' % (id, j + 1, w, h)),
                                      (255 * patch_clean).astype(np.uint8))
                    scipy.misc.imsave(os.path.join(dst_noise, '%s_%04d_%03d_%03d.bmp' % (id, j + 1, w, h)),
                                      (255 * patch_noise).astype(np.uint8))
                    cnt += 1
                    if cnt > 1023:
                        return
                    h += 32
                w += 32




if __name__ == '__main__':
    # dst = './data/original/'
    # if not os.path.exists(dst):
    #     os.makedirs(dst)
    # view('./data/challenge/', dst)
    # cut_single('./data/challenge/', './data/', 'noise')
    # cut('/data/hzw/npz/', './data/CUT_3mm/')
    # cut('/data/hzw/npz/', './data/CUT_1mm/')
    # cut_test('/data/hzw/npz/', './data/test_1mm/')
    cut_test('/data/hzw/npz/', './data/test_3mm/')
