#!/usr/bin/env python3

# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/main.py
#   + License: MIT
# [2016-08-05] Modifications for Inpainting: Brandon Amos (http://bamos.github.io)
#   + License: MIT

import os
import tensorflow as tf
from keras import backend as K
from model import GAN
from utils import pp


flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
# flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
# flags.DEFINE_float("learning_rate", 0.00005, "Learning rate of for adam [0.00005]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.0001]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
# flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("train_size", 1000000, "The size of train images [1000000]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 64, "The size of image to use [64]")
flags.DEFINE_integer("gpu_id", 0, "The id of gpu to use [0]")
flags.DEFINE_integer("c_dim", 1, "The channel of input images [1]")
flags.DEFINE_string("model_name", "dcgan", "The GAN model [dcgan]")
flags.DEFINE_string("ldct", "cut_noise", "Noise image dataset directory [cut_noise]")
flags.DEFINE_string("ndct", "cut_clean", "Clean image dataset directory [cut_clean]")
flags.DEFINE_string("checkpoint_dir", "ckpt", "Directory name to save the checkpoints [ckpt]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("train_dir", "train", "Directory name to save the training results [train]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
FLAGS = flags.FLAGS


def main(unuse_args):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    # if not os.path.exists(FLAGS.train_dir):
    #     os.makedirs(FLAGS.train_dir)

    # gpu = '/gpu:' + str(FLAGS.gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config = config) as sess:
        K.set_session(sess)
        gan = GAN(sess, model_name = FLAGS.model_name, image_size = FLAGS.image_size, batch_size = FLAGS.batch_size,
                      is_crop = False, c_dim = FLAGS.c_dim, checkpoint_dir = FLAGS.checkpoint_dir)

        if FLAGS.is_train:
            print('==========Training the model!==========')
            gan.train(FLAGS)
        else:
            print('==========Test the model!==========')
            gan.test(FLAGS)
            # gan.test_patch(FLAGS)


if __name__ == '__main__':
    tf.app.run()
