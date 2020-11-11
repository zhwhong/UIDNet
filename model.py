from __future__ import division
import os
import time
import math
import itertools
from glob import glob
import tensorflow as tf
import keras
from keras import backend as K
from six.moves import xrange
from ops import *
from utils import *
from skimage import measure
import scipy.io as sio
import shutil


SUPPORTED_EXTENSIONS = ["png", "bmp", "jpg"]


def dataset_files(root):
    """Returns a list of all image files in the given directory"""
    return list(itertools.chain.from_iterable(
        glob(os.path.join(root, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS))


def get_random_sample(shape, method = 'normal'):
    if method == 'uniform':
        sample_z = np.random.uniform(-1, 1, size = shape).astype(np.float32)
    elif method == 'random':
        sample_z = 2.0 * np.random.random(size = shape) - 1.0
    else:
        sample_z = np.random.normal(size = shape)
        sample_z = (sample_z - sample_z.min()) / (sample_z.max() - sample_z.min())
        sample_z = 2.0 * sample_z - 1.0
    return sample_z


def compute_mse_psnr_ssim(im_true, im_test, is_average = True):
    length = len(im_true.shape)
    mse = []
    psnr = []
    ssim = []
    if length == 2:
        mse.append(measure.compare_mse(im_true, im_test))
        psnr.append(measure.compare_psnr(im_true, im_test))
        ssim.append(measure.compare_ssim(im_true, im_test))
    elif length == 3:
        mse.append(measure.compare_mse(im_true, im_test))
        psnr.append(measure.compare_psnr(im_true, im_test))
        ssim.append(measure.compare_ssim(im_true, im_test, multichannel = True))
    elif length == 4:
        batch_size, w, h, c = im_true.shape
        for i in range(batch_size):
            mse.append(measure.compare_mse(im_true[i], im_test[i]))
            psnr.append(measure.compare_psnr(im_true[i], im_test[i]))
            ssim.append(measure.compare_ssim(im_true[i], im_test[i], multichannel = True))
    else:
        print("Error image shape!!!")

    if is_average:
        return np.mean(mse), np.mean(psnr), np.mean(ssim)
    else:
        return mse, psnr, ssim


class GAN(object):
    def __init__(self, sess, model_name = 'dcgan', image_size = 64, is_crop = False,
                 batch_size = 64, sample_size = 64, df_dim = 64, c_dim = 1,
                 checkpoint_dir = None, lam = 10.0, use_image_gradient = 'G1'):

        # Currently, image size must be a (power of 2) and (8 or higher).
        assert(image_size & (image_size - 1) == 0 and image_size >= 8)

        # Initialize the parameters
        self.sess = sess
        self.image_size = image_size
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.df_dim = df_dim
        self.c_dim = c_dim
        self.checkpoint_dir = checkpoint_dir
        self.lam = lam
        self.image_shape = [image_size, image_size, c_dim]
        self.model_name = model_name
        self.use_image_gradient = use_image_gradient

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bns = [batch_norm(name = 'd_bn{}'.format(i,)) for i in range(3)]

        # build the model
        self.build_model()


    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, name = 'is_training')

        # Generator and Discriminator input
        self.ndct = tf.placeholder(tf.float32, [None] + self.image_shape, name = 'NDCT_images')
        self.ldct = tf.placeholder(tf.float32, [None] + self.image_shape, name = 'LDCT_images')
        self.z = tf.placeholder(tf.float32, [None] + self.image_shape, name = 'z')
        # self.z_sum = tf.summary.histogram("z", self.z)

        # DNN model input
        # self.virtual_ldct = tf.placeholder(tf.float32, [None] + self.image_shape, name = 'Virutal_LDCT_images')
        self.virtual_ldct = tf.placeholder(tf.float32, [None, None, None, self.c_dim], name = 'Virutal_LDCT_images')

        # Generator model and DNN model
        self.g_model = self.generator(self.z, self.ndct)    # self.ndct as a conditional image, it's optional
        self.dnn_model = self.dnn(self.virtual_ldct)

        self.g_noise = self.g_model([self.z, self.ndct])
        self.G = self.g_noise + self.ndct
        # self.G = tf.clip_by_value(self.G, -1.0, 1.0)
        # self.G_sum = tf.summary.image("G", self.G)

        self.dnn_pred = self.dnn_model(self.G)
        self.out = self.G - self.dnn_pred


        # Calculate the image gradient - the sharpening technique
        def diff(x):
            if self.use_image_gradient == 'G1':
                g1 = x - tf.nn.avg_pool(x, ksize = [1, 3, 3, 1], strides = [1, 1, 1, 1], padding = 'SAME')
                g = tf.concat([x, g1], axis = 3)
            elif self.use_image_gradient == 'G2':
                g1 = x - tf.nn.avg_pool(x, ksize = [1, 3, 3, 1], strides =[ 1, 1, 1, 1], padding = 'SAME')
                g2 = x - tf.nn.avg_pool(x, ksize = [1, 5, 5, 1], strides = [1, 1, 1, 1], padding = 'SAME')
                g = tf.concat([x, g1, g2], axis = 3)
            elif self.use_image_gradient == 'G3':
                g1 = x - tf.nn.avg_pool(x, ksize = [1, 3, 3, 1], strides = [1, 1, 1, 1], padding = 'SAME')
                g2 = x - tf.nn.avg_pool(x, ksize = [1, 5, 5, 1], strides = [1, 1, 1, 1], padding = 'SAME')
                g3 = x - tf.nn.avg_pool(x, ksize = [1, 7, 7, 1], strides = [1, 1, 1, 1], padding = 'SAME')
                g = tf.concat([x, g1, g2, g3], axis = 3)
            elif self.use_image_gradient == 'G4':
                g1 = x - tf.nn.avg_pool(x, ksize = [1, 3, 3, 1], strides = [1, 1, 1, 1], padding = 'SAME')
                g2 = x - tf.nn.avg_pool(x, ksize = [1, 5, 5, 1], strides = [1, 1, 1, 1], padding = 'SAME')
                g3 = x - tf.nn.avg_pool(x, ksize = [1, 7, 7, 1], strides = [1, 1, 1, 1], padding = 'SAME')
                g4 = x - tf.nn.avg_pool(x, ksize = [1, 11, 11, 1], strides = [1, 1, 1, 1], padding = 'SAME')
                g = tf.concat([x, g1, g2, g3, g4], axis = 3)
            elif self.use_image_gradient == 'G5':
                g1 = x - tf.nn.avg_pool(x, ksize = [1, 3, 3, 1], strides = [1, 1, 1, 1], padding = 'SAME')
                g2 = x - tf.nn.avg_pool(x, ksize = [1, 5, 5, 1], strides = [1, 1, 1, 1], padding = 'SAME')
                g3 = x - tf.nn.avg_pool(x, ksize = [1, 7, 7, 1], strides = [1, 1, 1, 1], padding = 'SAME')
                gg1 = g1 - tf.nn.avg_pool(g1, ksize = [1, 3, 3, 1], strides = [1, 1, 1, 1], padding = 'SAME')
                gg2 = g2 - tf.nn.avg_pool(g2, ksize = [1, 5, 5, 1], strides = [1, 1, 1, 1], padding = 'SAME')
                g = tf.concat([x, g1, g2, g3, gg1, gg2], axis = 3)
            else:
                g = x
            return g


        self.noisy_real = diff(self.ldct)
        self.noisy_fake = diff(self.G)

        # Discriminator logits and labels
        self.D_real, self.D_real_logits = self.discriminator(self.noisy_real)
        self.D_fake, self.D_fake_logits = self.discriminator(self.noisy_fake, reuse = True)

        # self.d_real_sum = tf.summary.histogram("d_real", self.D_real)
        # self.d_fake_sum = tf.summary.histogram("d_fake", self.D_fake)

        # Loss function
        if self.model_name == 'dcgan':
            # 1. dcgan generator loss function
            self.g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_fake_logits,
                                                        labels = tf.ones_like(self.D_fake)))

            # 2. dcgan discriminator loss function
            self.d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_real_logits,
                                                        labels = tf.ones_like(self.D_real)))
            self.d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_fake_logits,
                                                        labels = tf.zeros_like(self.D_fake)))
            self.d_loss = self.d_loss_real + self.d_loss_fake

        elif self.model_name == 'wgan-gp':
            # wgan generator loss function
            self.g_loss = -tf.reduce_mean(self.D_fake_logits)

            # wgan discriminator loss function
            epsilon = tf.random_uniform([], 0.0, 1.0)
            interpolates = epsilon * self.ldct + (1 - epsilon) * self.G
            interpolates = diff(interpolates)
            gradients = tf.gradients(self.discriminator(interpolates, reuse = True)[-1], [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis = 1))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

            # wd = tf.reduce_mean(self.D_real_logits) - tf.reduce_mean(self.D_fake_logits)
            # self.d_loss = -wd + self.lam * gradient_penalty

            self.d_loss_real = -tf.reduce_mean(self.D_real_logits)
            # self.d_loss_fake = tf.reduce_mean(self.D_fake_logits)
            self.d_loss_fake = -self.g_loss
            self.d_loss_gradient = gradient_penalty
            self.d_loss = self.d_loss_real + self.d_loss_fake + self.lam * self.d_loss_gradient

        # 3. dnn model loss function
        self.dnn_loss = (1.0 / self.batch_size) * tf.nn.l2_loss(self.out - self.ndct)  # L2 Loss
        # self.dnn_loss = tf.reduce_mean(tf.square(self.out - self.ndct))  # MSE Loss
        # self.dnn_loss = tf.reduce_mean(tf.abs(self.out - self.ndct))  # L1 Loss

        # The train and validation loss curve in tensorboard
        self.g_loss_sum = tf.summary.scalar("train/g_loss", self.g_loss)
        #self.d_loss_real_sum = tf.summary.scalar("train/d_loss_real", self.d_loss_real)
        #self.d_loss_fake_sum = tf.summary.scalar("train/d_loss_fake", self.d_loss_fake)
        self.d_loss_sum = tf.summary.scalar("train/d_loss", self.d_loss)
        self.dnn_loss_sum = tf.summary.scalar("train/dnn_loss", self.dnn_loss)

        self.g_loss_val_sum = tf.summary.scalar("val/g_loss", self.g_loss)
        #self.d_loss_real_val_sum = tf.summary.scalar("val/d_loss_real", self.d_loss_real)
        #self.d_loss_fake_val_sum = tf.summary.scalar("val/d_loss_fake", self.d_loss_fake)
        self.d_loss_val_sum = tf.summary.scalar("val/d_loss", self.d_loss)
        self.dnn_loss_val_sum = tf.summary.scalar("val/dnn_loss", self.dnn_loss)

        #if self.model_name == 'wgan-gp':
        #    self.d_loss_gradient_sum = tf.summary.scalar("train/d_loss_gradient", self.d_loss_gradient)
        #    self.d_loss_gradient_val_sum = tf.summary.scalar("val/d_loss_gradient", self.d_loss_gradient)


        # Training process control
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.dnn_vars = [var for var in t_vars if 'dnn' in var.name]

        # Save the model checkpoint
        # self.saver = tf.train.Saver(max_to_keep = 3)
        self.saver = tf.train.Saver(max_to_keep = 10)


    def train(self, config):
        data_ndct = dataset_files(config.ndct)
        data_ldct = dataset_files(config.ldct)

        assert (len(data_ndct) > 0)
        assert (len(data_ldct) > 0)

        # np.random.shuffle(data_ndct)
        # np.random.shuffle(data_ldct)

        """
        d_optim = tf.train.RMSPropOptimizer(config.learning_rate) \
                        .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.RMSPropOptimizer(config.learning_rate) \
                        .minimize(self.g_loss, var_list=self.g_vars)
        dnn_optim = tf.train.RMSPropOptimizer(config.learning_rate) \
                        .minimize(self.dnn_loss, var_list=self.dnn_vars)
        """

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1 = config.beta1) \
                          .minimize(self.d_loss, var_list = self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1 = config.beta1) \
                          .minimize(self.g_loss, var_list = self.g_vars)
        dnn_optim = tf.train.AdamOptimizer(config.learning_rate, beta1 = config.beta1) \
                          .minimize(self.dnn_loss, var_list = self.dnn_vars)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        """
        self.g_sum = tf.summary.merge([self.z_sum, self.d_fake_sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.z_sum, self.d_real_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.dnn_sum = tf.summary.merge([self.z_sum, self.dnn_loss_sum])
        """

        #self.g_sum = tf.summary.merge([self.d_loss_fake_sum, self.g_loss_sum])
        #self.dnn_sum = tf.summary.merge([self.dnn_loss_sum])
        self.g_sum = self.g_loss_sum
        self.dnn_sum = self.dnn_loss_sum
        self.d_sum = self.d_loss_sum
        self.val_sum = tf.summary.merge([self.g_loss_val_sum, self.d_loss_val_sum, self.dnn_loss_val_sum])

        """
        if self.model_name == 'wgan-gp':
            self.d_sum = tf.summary.merge([self.d_loss_real_sum, self.d_loss_gradient_sum, self.d_loss_sum])
            self.val_sum = tf.summary.merge([self.g_loss_val_sum, self.d_loss_fake_val_sum, self.d_loss_real_val_sum,
                                             self.d_loss_gradient_val_sum, self.d_loss_val_sum, self.dnn_loss_val_sum])
        else:
            self.d_sum = tf.summary.merge([self.d_loss_real_sum, self.d_loss_sum])
            self.val_sum = tf.summary.merge([self.g_loss_val_sum, self.d_loss_fake_val_sum, self.d_loss_real_val_sum,
                                             self.d_loss_val_sum, self.dnn_loss_val_sum])
        """

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        # Random noise image sample of shape [sample_size, 64, 64, 1], range -1~1, method = 'normal','uniform','random'
        sample_z = get_random_sample(([self.sample_size] + self.image_shape), method = 'uniform')

        # sample_index = np.random.choice(len(data_ndct), self.sample_size, replace = False)
        # print("Random sample validation batch: ")
        # print(sample_index)

        # NDCT image sample
        # sample_files_ndct = data_ndct[0:self.sample_size]
        # sample_files_ndct = [data_ndct[i] for i in sample_index]
        sample_files_ndct = sorted(dataset_files('./data/test_1mm/cut_clean/'))[:self.sample_size]
        sample_ndct = [get_image(i, self.image_size, is_crop = self.is_crop) for i in sample_files_ndct]
        sample_ndct = np.array(sample_ndct).astype(np.float32)

        # LDCT image sample
        # sample_files_ldct = data_ldct[0:self.sample_size]
        sample_files_ldct = sorted(dataset_files('./data/test_1mm/cut_noise/'))[:self.sample_size]
        # sample_files_ldct = [data_ldct[i] for i in sample_index]
        sample_ldct = [get_image(i, self.image_size, is_crop = self.is_crop) for i in sample_files_ldct]
        sample_ldct = np.array(sample_ldct).astype(np.float32)

        save_images(sample_ndct, [8, 8], './samples/ndct_validation.png')
        save_images(sample_ldct, [8, 8], './samples/ldct_validation.png')

        # Start training
        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print("""\n============\nAn existing model was found in the checkpoint directory!\n============\n""")
        else:
            print("""\n============\nNo existing model found! Initializing a new one.\n============\n""")

        # Rest data for training
        # data_ndct = data_ndct[self.sample_size:]
        # data_ldct = data_ldct[self.sample_size:]

        #data_ndct = [data_ndct[i] for i in np.arange(len(data_ndct)) if i not in sample_index]
        #data_ldct = [data_ldct[i] for i in np.arange(len(data_ldct)) if i not in sample_index]
        batch_idxs = min(len(data_ndct), len(data_ldct), config.train_size) // self.batch_size

        max_psnr = 0.0

        for epoch in xrange(config.epoch):
            np.random.shuffle(data_ndct)
            np.random.shuffle(data_ldct)

            for idx in xrange(0, batch_idxs):
                # Normal Does CT data (real clean data)
                batch_files_ndct = data_ndct[idx * config.batch_size : (idx + 1) * config.batch_size]
                batch_ndct = [get_image(i, self.image_size, is_crop = self.is_crop) for i in batch_files_ndct]
                batch_ndct = np.array(batch_ndct).astype(np.float32)

                # Low Does CT data (real noise data)
                batch_files_ldct = data_ldct[idx * config.batch_size : (idx + 1) * config.batch_size]
                batch_ldct = [get_image(i, self.image_size, is_crop = self.is_crop) for i in batch_files_ldct]
                batch_ldct = np.array(batch_ldct).astype(np.float32)

                # Random noise
                batch_z = get_random_sample(([config.batch_size] + self.image_shape), method = 'uniform')

                # Update D network
                #d_iters = 3

                #if (idx % d_iters) != (d_iters - 1):
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                    feed_dict={ self.ndct: batch_ndct, self.ldct: batch_ldct, self.z: batch_z, self.is_training: True, K.learning_phase(): 1 })
                self.writer.add_summary(summary_str, counter)
                #    continue

                # Update G network
                _, summary_str, _ = self.sess.run([g_optim, self.g_sum, self.g_model.updates],
                    feed_dict={ self.z: batch_z, self.ndct: batch_ndct, self.is_training: True, K.learning_phase(): 1 })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                # _, summary_str, _ = self.sess.run([g_optim, self.g_sum, self.g_model.updates],
                #     feed_dict={ self.z: batch_z, self.ndct: batch_ndct, self.is_training: True, K.learning_phase(): 1 })
                # self.writer.add_summary(summary_str, counter)

                # Update DNN network
                G_ = self.sess.run(self.G, feed_dict = {self.z: batch_z, self.ndct: batch_ndct, self.is_training: False, K.learning_phase(): 0})
                _, summary_str, _ = self.sess.run([dnn_optim, self.dnn_sum , self.dnn_model.updates],
                    feed_dict = { self.z: batch_z, self.ndct: batch_ndct, self.virtual_ldct: G_, self.is_training: True, K.learning_phase(): 1 })
                self.writer.add_summary(summary_str, counter)

                # Generator and Discriminator Loss
                errD, errG, errDNN = self.sess.run([self.d_loss, self.g_loss, self.dnn_loss],
                    feed_dict={ self.z: batch_z, self.ndct: batch_ndct, self.ldct: batch_ldct, self.is_training: False, K.learning_phase(): 0 })

                counter += 1
                print("Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}, dnn_loss: {:.8f}".format(
                    epoch, idx, batch_idxs, time.time() - start_time, errD, errG, errDNN))

                # training set output
                """
                if np.mod(counter, 50) == 1:
                    virtual_ldcts, noises, dnn_pred, out = self.sess.run([self.G, self.g_noise, self.dnn_pred, self.out],
                        feed_dict={self.z: batch_z, self.ndct: batch_ndct, self.is_training: False, K.learning_phase(): 0})

                    save_images(batch_ndct, [8, 8], './train/ndct_{:02d}_{:04d}.png'.format(epoch, idx))
                    save_images(batch_ldct, [8, 8], './train/ldct_{:02d}_{:04d}.png'.format(epoch, idx))

                    save_images(virtual_ldcts, [8, 8], './train/virtual_ldct_{:02d}_{:04d}.png'.format(epoch, idx))
                    save_images(noises, [8, 8], './train/g_noise_{:02d}_{:04d}.png'.format(epoch, idx))
                    save_images(dnn_pred, [8, 8], './train/dnn_noise_{:02d}_{:04d}.png'.format(epoch, idx))
                    save_images(out, [8, 8], './train/denoised_{:02d}_{:04d}.png'.format(epoch, idx))

                    print("noise: ", noises.min(), noises.max(), noises.mean())
                    print("ndct: ", batch_ndct.min(), batch_ndct.max(), batch_ndct.mean())
                    print("generate_ldct: ", virtual_ldcts.min(), virtual_ldcts.max(), virtual_ldcts.mean())
                """

                # validation loss curve
                if np.mod(counter, 100) == 1:
                    summary_str = self.sess.run(self.val_sum,
                                                feed_dict = {self.ndct: sample_ndct, self.ldct: sample_ldct,
                                                           self.z: sample_z, self.is_training: False, K.learning_phase(): 0})
                    self.writer.add_summary(summary_str, counter)

                # validation set output
                if np.mod(counter, 300) == 1:
                    virtual_ldcts, noises, dnn_pred, out, d_loss, g_loss, dnn_loss = self.sess.run(
                        [self.G, self.g_noise, self.dnn_pred, self.out, self.d_loss, self.g_loss, self.dnn_loss],
                        feed_dict = {self.z: sample_z, self.ndct: sample_ndct, self.ldct: sample_ldct, self.is_training: False, K.learning_phase(): 0})

                    save_images(virtual_ldcts, [8, 8], './samples/virtual_ldct_{:02d}_{:04d}.png'.format(epoch, idx))
                    save_images(noises, [8, 8], './samples/g_noise_{:02d}_{:04d}.png'.format(epoch, idx))
                    save_images(dnn_pred, [8, 8], './samples/dnn_noise_{:02d}_{:04d}.png'.format(epoch, idx))
                    save_images(out, [8, 8], './samples/denoised_{:02d}_{:04d}.png'.format(epoch, idx))
                    print("[Validation] d_loss: {:.8f}, g_loss: {:.8f}, dnn_loss: {:.8f}".format(d_loss, g_loss, dnn_loss))

                if np.mod(counter, 1000) == 2:
                    self.save(config.checkpoint_dir, counter)
                    print("[counter = %d] save a new model checkpoint successfully!" % (counter,))

                    dnn_pred_real = self.sess.run(self.dnn_model(self.virtual_ldct),
                                                  feed_dict = {self.virtual_ldct: sample_ldct, self.is_training: False, K.learning_phase(): 0})
                    denoised_real = sample_ldct - dnn_pred_real
                    cur_mse, cur_psnr, cur_ssim = compute_mse_psnr_ssim(denormalize(sample_ndct), denormalize(denoised_real))
                    print("[counter = {:02d}] Denoised  out: MSE = {:.8f}, PSNR = {:.8f}, SSIM = {:.8f}".format(counter, cur_mse, cur_psnr, cur_ssim))

                    if cur_psnr >= max_psnr:
                        max_psnr = cur_psnr
                        shutil.copytree('ckpt', 'ckpt_%d'%(counter, ))
                        print("[counter = %d] find a new model with higher psnr!" % (counter,))


    def discriminator(self, image, reuse = False):
        if self.model_name == "dcgan":
            with tf.variable_scope("discriminator") as scope:
                if reuse:
                    scope.reuse_variables()
                # input : (batch_size, 64, 64, 1)
                h0 = lrelu(conv2d(image, self.df_dim, name = 'd_h0_conv'))
                # h0: (batch_size, 32, 32, 64)
                h1 = lrelu(self.d_bns[0](conv2d(h0, self.df_dim*2, name = 'd_h1_conv'), self.is_training))
                # h1: (batch_size, 16, 16, 128)
                h2 = lrelu(self.d_bns[1](conv2d(h1, self.df_dim*4, name = 'd_h2_conv'), self.is_training))
                # h2: (batch_size, 8, 8, 256)
                h3 = lrelu(self.d_bns[2](conv2d(h2, self.df_dim*8, name = 'd_h3_conv'), self.is_training))
                # h3: (batch_size, 4, 4, 512)
                h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h4_lin')
                # h4: (batch_size, 8192)

                # print(h0.shape, h1.shape, h2.shape, h3.shape, h4.shape)
                return tf.nn.sigmoid(h4), h4

        elif self.model_name == "wgan-gp":
            with tf.variable_scope("discriminator") as scope:
                if reuse:
                    scope.reuse_variables()
                # input : (batch_size, 64, 64, 1)
                h0 = lrelu(conv2d(image, self.df_dim, name = 'd_h0_conv'))
                # h0: (batch_size, 32, 32, 64)
                h1 = lrelu(conv2d(h0, self.df_dim*2, name = 'd_h1_conv'))
                # h1: (batch_size, 16, 16, 128)
                h2 = lrelu(conv2d(h1, self.df_dim*4, name = 'd_h2_conv'))
                # h2: (batch_size, 8, 8, 256)
                h3 = lrelu(conv2d(h2, self.df_dim*8, name = 'd_h3_conv'))
                # h3: (batch_size, 4, 4, 512)
                h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h4_lin')
                # h4: (batch_size, 8192)

                # print(h0.shape, h1.shape, h2.shape, h3.shape, h4.shape)
                return tf.nn.sigmoid(h4), h4
        else:
            pass


    """
    def generator_mini(self, z, y = None):
        with tf.variable_scope("generator") as scope:
            input_a = keras.models.Input(tensor = z)
            input_b = None
            input = input_a
            if y is not None:
                input_b = keras.models.Input(tensor = y)
                input = keras.layers.concatenate([input_a, input_b], axis = 3)

            conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
            conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)  # [batch_size, 32, 32, 32]

            conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
            conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
            drop2 = keras.layers.Dropout(0.5)(conv2)  # [batch_size, 32, 32, 64]

            up3 = keras.layers.Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                keras.layers.UpSampling2D(size = (2, 2))(drop2))  # [batch_size, 64, 64, 32]
            merge = keras.layers.concatenate([conv1, up3], axis=3)
            conv3 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge)
            conv3 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)

            conv3 = keras.layers.Conv2D(self.c_dim, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
            output = keras.layers.Conv2D(self.c_dim, 1, activation='tanh')(conv3)  # # [batch_size, 64, 64, c_dim]

            if y is not None:
                model = keras.models.Model([input_a, input_b], output)
            else:
                model = keras.models.Model(input_a, output)
            # model.summary()
            return model
    """


    ### median U-Net
    def generator(self, z, y = None):
        with tf.variable_scope("generator") as scope:
            input_a = keras.models.Input(tensor=z)
            input_b = None
            input = input_a
            if y is not None:
                input_b = keras.models.Input(tensor=y)
                input = keras.layers.concatenate([input_a, input_b], axis=3)

            conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
            conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)  # [batch_size, 32, 32, 32]

            conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
            conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
            pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)  # [batch_size, 16, 16, 64]

            conv3 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
            conv3 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
            drop3 = keras.layers.Dropout(0.5)(conv3)                    # [batch_size, 16, 16, 128]

            up4 = keras.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                keras.layers.UpSampling2D(size=(2, 2))(drop3))  # [batch_size, 32, 32, 64]
            merge4 = keras.layers.concatenate([conv2, up4], axis=3)
            conv4 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge4)
            conv4 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

            up5 = keras.layers.Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                keras.layers.UpSampling2D(size=(2, 2))(conv4))  # [batch_size, 64, 64, 32]
            merge5 = keras.layers.concatenate([conv1, up5], axis=3)
            conv5 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
            conv5 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

            conv5 = keras.layers.Conv2D(self.c_dim, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
            output = keras.layers.Conv2D(self.c_dim, 1, activation='tanh')(conv5)

            # output = keras.layers.Conv2D(self.c_dim, 1, activation='tanh', padding='same', kernel_initializer='he_normal')(conv5)

            if y is not None:
                model = keras.models.Model([input_a, input_b], output)
            else:
                model = keras.models.Model(input_a, output)
            # model.summary()
            return model


    """
    ### median U-Net 2
    def generator3(self, z, y = None):
        with tf.variable_scope("generator") as scope:
            input_a = keras.models.Input(tensor=z)
            input_b = None
            input = input_a
            if y is not None:
                input_b = keras.models.Input(tensor=y)
                input = keras.layers.concatenate([input_a, input_b], axis=3)

            conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
            conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)  # [batch_size, 32, 32, 32]

            conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
            conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
            pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)  # [batch_size, 16, 16, 64]

            conv3 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
            conv3 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
            drop3 = keras.layers.Dropout(0.5)(conv3)
            pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop3)  # [batch_size, 8, 8, 128]

            conv4 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
            conv4 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
            drop4 = keras.layers.Dropout(0.5)(conv4)  # [batch_size, 8, 8, 256]

            up5 = keras.layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                keras.layers.UpSampling2D(size=(2, 2))(drop4))  # [batch_size, 16, 16, 128]
            merge5 = keras.layers.concatenate([drop3, up5], axis=3)
            conv5 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
            conv5 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

            up6 = keras.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                keras.layers.UpSampling2D(size=(2, 2))(conv5))  # [batch_size, 32, 32, 64]
            merge6 = keras.layers.concatenate([conv2, up6], axis=3)
            conv6 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
            conv6 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

            up7 = keras.layers.Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                keras.layers.UpSampling2D(size=(2, 2))(conv6))  # [batch_size, 64, 64, 32]
            merge7 = keras.layers.concatenate([conv1, up7], axis=3)
            conv7 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
            conv7 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

            conv7 = keras.layers.Conv2D(self.c_dim, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
            output = keras.layers.Conv2D(self.c_dim, 1, activation='tanh')(conv7)

            if y is not None:
                model = keras.models.Model([input_a, input_b], output)
            else:
                model = keras.models.Model(input_a, output)
            # model.summary()
            return model
    """


    def dnn(self, x):
        with tf.variable_scope("dnn") as scope:
            input = keras.models.Input(tensor = x)
            # 1st layer, Conv + relu
            output = keras.layers.Conv2D(filters = 64, kernel_size=(3, 3), strides = (1, 1), padding = 'same')(input)
            output = keras.layers.Activation('relu')(output)
            # 15 layers, Conv + BN + relu
            for i in range(15):
                output = keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(output)
                output = keras.layers.BatchNormalization(axis = -1, epsilon = 1e-3)(output)
                output = keras.layers.Activation('relu')(output)
            # last layer, Conv
            output = keras.layers.Conv2D(filters = self.c_dim, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(output)
            # output = keras.layers.Conv2D(filters = self.c_dim, kernel_size = (3, 3), activation = 'relu', strides = (1, 1), padding = 'same')(output)
            # output = keras.layers.Conv2D(self.c_dim, 1, activation='tanh')(output)
            # output = keras.layers.Subtract()([input, output])  # input - noise
            model = keras.models.Model(inputs = input, outputs = output)
            return model


    """
    def dnn2(self, x):
        with tf.variable_scope("dnn") as scope:
            input = keras.models.Input(tensor = x)

            conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
            conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            pool1 = keras.layers.MaxPooling2D(pool_size = (2, 2))(conv1)  # [batch_size, 32, 32, 32]

            conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
            conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
            drop2 = keras.layers.Dropout(0.5)(conv2)  # [batch_size, 32, 32, 64]

            up3 = keras.layers.Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                keras.layers.UpSampling2D(size=(2, 2))(drop2))  # [batch_size, 64, 64, 32]
            merge = keras.layers.concatenate([conv1, up3], axis=3)
            conv3 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge)
            conv3 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)

            conv3 = keras.layers.Conv2D(self.c_dim, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
            output = keras.layers.Conv2D(self.c_dim, 1, activation='tanh')(conv3)

            model = keras.models.Model(input, output)
            # model.summary()
            return model
    """
    """
    def dnn_unet(self, x):
        with tf.variable_scope("dnn") as scope:
            input = keras.models.Input(tensor = x)

            conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
            conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)  # [batch_size, 32, 32, 32]

            conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
            conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
            pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)  # [batch_size, 16, 16, 64]

            conv3 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
            conv3 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
            drop3 = keras.layers.Dropout(0.5)(conv3)
            pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop3)  # [batch_size, 8, 8, 128]

            conv4 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
            conv4 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
            drop4 = keras.layers.Dropout(0.5)(conv4)  # [batch_size, 8, 8, 256]

            up5 = keras.layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                keras.layers.UpSampling2D(size=(2, 2))(drop4))  # [batch_size, 16, 16, 128]
            merge5 = keras.layers.concatenate([drop3, up5], axis=3)
            conv5 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
            conv5 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

            up6 = keras.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                keras.layers.UpSampling2D(size=(2, 2))(conv5))  # [batch_size, 32, 32, 64]
            merge6 = keras.layers.concatenate([conv2, up6], axis=3)
            conv6 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
            conv6 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

            up7 = keras.layers.Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                keras.layers.UpSampling2D(size=(2, 2))(conv6))  # [batch_size, 64, 64, 32]
            merge7 = keras.layers.concatenate([conv1, up7], axis=3)
            conv7 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
            conv7 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

            conv7 = keras.layers.Conv2D(self.c_dim, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
            output = keras.layers.Conv2D(self.c_dim, 1, activation='tanh')(conv7)

            model = keras.models.Model(input, output)
            # model.summary()
            return model
    """


    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name), global_step = step)


    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False


    def test_dnd(self, config):
        if self.load(self.checkpoint_dir):
            print(" [*] Load ckeckpoint successfully!!!")
        else:
            print(" [!] Load checkpoint failed...")

        if not os.path.exists('test'):
            os.makedirs('test')

        data_ldct = sorted(dataset_files(config.ldct))
        assert (len(data_ldct) > 0)

        length = len(data_ldct)
        print('test cases: ', length)

        # submit
        out_folder = os.path.join('./', "bundled/")
        try:
            os.mkdir(out_folder)
        except:
            pass
        israw = False
        eval_version = "1.0"


        for i in range(length):
            print("============== [id = %d/%d] ==============" % (i + 1, length))

            ldct_image = [get_image(data_ldct[i], self.image_size, is_crop = self.is_crop)]
            ldct_image = np.array(ldct_image).astype(np.float32)

            dnn_pred_real = self.sess.run(self.dnn_model(self.virtual_ldct),
                                          feed_dict = {self.virtual_ldct: ldct_image, self.is_training: False, K.learning_phase(): 0})
            output_real = ldct_image - dnn_pred_real

            id = i // 20
            bb = i % 20

            save_image(dnn_pred_real[0], './test/noise_{:02d}_{:02d}.png'.format(id + 1, bb + 1))
            save_image(output_real[0], './test/denoised_{:02d}_{:02d}.png'.format(id + 1, bb + 1))

            if bb == 0:
                Idenoised = np.zeros((20,), dtype = np.object)

            Idenoised[bb] = np.clip(inverse_transform(output_real[0]), 0.0, 1.0)

            if bb == 19:
                filename = '%04d.mat' % (id + 1)
                sio.savemat(os.path.join(out_folder, filename),
                            {"Idenoised": Idenoised,
                             "israw": israw,
                             "eval_version": eval_version},
                            )
                print("[image id = %02d] save %s successfully!" % (id + 1, filename))


    def test_sidd(self, config):
        if self.load(self.checkpoint_dir):
            print(" [*] Load ckeckpoint successfully!!!")
        else:
            print(" [!] Load checkpoint failed...")

        if not os.path.exists('test'):
            os.makedirs('test')

        data_ldct = sorted(dataset_files(config.ldct))
        assert (len(data_ldct) > 0)

        length = len(data_ldct)
        print('test cases: ', length)

        mat = []

        for i in range(length):
            print("============== [id = %d/%d] ==============" % (i + 1, length))

            ldct_image = [get_image(data_ldct[i], self.image_size, is_crop = self.is_crop)]
            ldct_image = np.array(ldct_image).astype(np.float32)

            dnn_pred_real = self.sess.run(self.dnn_model(self.virtual_ldct),
                                          feed_dict = {self.virtual_ldct: ldct_image, self.is_training: False, K.learning_phase(): 0})
            output_real = ldct_image - dnn_pred_real

            save_image(dnn_pred_real[0], './test/noise_{:04d}.png'.format(i + 1))
            save_image(output_real[0], './test/denoised_{:04d}.png'.format(i + 1))

            denoised_image = 255.0 * inverse_transform(output_real[0])
            denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)
            mat.append(denoised_image)

        mat = np.array(mat)
        mat = np.reshape(mat, [40, 32, 256, 256, 3])
        sio.savemat('Denoised.mat', {'results': mat})


    def test_patch(self, config):
        if self.load(self.checkpoint_dir):
            print(" [*] Load ckeckpoint successfully!!!")
        else:
            print(" [!] Load checkpoint failed...")

        if not os.path.exists('test'):
            os.makedirs('test')

        # data_ndct = sorted(dataset_files(config.ndct))
        # data_ldct = sorted(dataset_files(config.ldct))

        data_ndct = dataset_files(config.ndct)[:4*config.batch_size]
        data_ldct = dataset_files(config.ldct)[:4*config.batch_size]

        # data_ndct = dataset_files(config.ndct)
        # data_ldct = dataset_files(config.ldct)

        assert (len(data_ndct) > 0)
        assert (len(data_ldct) > 0)

        mse_before = []
        psnr_before = []
        ssim_before = []
        mse_after = []
        psnr_after = []
        ssim_after = []
        mse_real_before = []
        psnr_real_before = []
        ssim_real_before = []
        mse_real_after = []
        psnr_real_after = []
        ssim_real_after = []

        batch_idxs = min(len(data_ndct), len(data_ldct)) // self.batch_size

        print('test dataset size: ', len(data_ndct))
        print('test batch idxs: ', batch_idxs)

        for idx in xrange(0, batch_idxs):
            print("============== [batch = %d/%d] ==============" % (idx+1, batch_idxs))
            # Normal Does CT data (real clean data)
            batch_files_ndct = data_ndct[idx * config.batch_size: (idx + 1) * config.batch_size]
            batch_ndct = [get_image(i, self.image_size, is_crop=self.is_crop) for i in batch_files_ndct]
            batch_ndct = np.array(batch_ndct).astype(np.float32)

            # Low Does CT data (real noise data)
            batch_files_ldct = data_ldct[idx * config.batch_size: (idx + 1) * config.batch_size]
            batch_ldct = [get_image(i, self.image_size, is_crop=self.is_crop) for i in batch_files_ldct]
            batch_ldct = np.array(batch_ldct).astype(np.float32)

            # Random noise image sample of shape [sample_size, 64, 64, 1], range -1~1, method = 'normal','uniform','random'
            batch_z = get_random_sample(([config.batch_size] + self.image_shape), method='uniform')

            save_images(batch_ndct, [8, 8], './test/ndct_test_{:04d}.png'.format(idx))
            save_images(batch_ldct, [8, 8], './test/ldct_test_{:04d}.png'.format(idx))

            virtual_ldct, g_noise, dnn_pred, output = self.sess.run(
                [self.G, self.g_noise, self.dnn_pred, self.out],
                feed_dict = {self.z: batch_z, self.ndct: batch_ndct, self.is_training: False, K.learning_phase(): 0})

            # g_noise = self.sess.run(self.g_model([self.z, self.ndct]),
            #             feed_dict = {self.z: batch_z, self.ndct: batch_ndct, self.is_training: False, K.learning_phase(): 0})
            # virtual_ldct = g_noise + batch_ndct
            save_images(g_noise, [8, 8], './test/g_noise_{:04d}.png'.format(idx))
            save_images(virtual_ldct, [8, 8], './test/virtual_ldct_{:04d}.png'.format(idx))

            # test the denoising performance on virtual ldct
            # print("============== Test on virtual ldct ==============")
            # dnn_pred = self.sess.run(self.dnn_model(self.virtual_ldct),
            #             feed_dict = {self.virtual_ldct: virtual_ldct, self.is_training: False, K.learning_phase(): 0})
            # output = virtual_ldct - dnn_pred
            save_images(dnn_pred, [8, 8], './test/dnn_noise_{:04d}.png'.format(idx))
            save_images(output, [8, 8], './test/denoised_{:04d}.png'.format(idx))

            # print("1. psnr on normalized results 0~1")
            """
            mse, psnr, ssim = compute_mse_psnr_ssim(inverse_transform(batch_ndct), inverse_transform(virtual_ldct), is_average = False)
            print("[Virtual  LDCT] MSE = {:.8f}, PSNR = {:.8f}, SSIM = {:.8f}".format(np.mean(mse), np.mean(psnr), np.mean(ssim)))
            mse_before.extend(mse)
            psnr_before.extend(psnr)
            ssim_before.extend(ssim)

            mse, psnr, ssim = compute_mse_psnr_ssim(inverse_transform(batch_ndct), inverse_transform(output), is_average = False)
            print("[Denoised  out] MSE = {:.8f}, PSNR = {:.8f}, SSIM = {:.8f}".format(np.mean(mse), np.mean(psnr), np.mean(ssim)))
            mse_after.extend(mse)
            psnr_after.extend(psnr)
            ssim_after.extend(ssim)
            """

            # print("2. psnr on denormalized results 0~255")
            mse, psnr, ssim = compute_mse_psnr_ssim(denormalize(batch_ndct), denormalize(virtual_ldct), is_average = False)
            print("[Virtual  LDCT] MSE = {:.8f}, PSNR = {:.8f}, SSIM = {:.8f}".format(np.mean(mse), np.mean(psnr), np.mean(ssim)))
            mse_before.extend(mse)
            psnr_before.extend(psnr)
            ssim_before.extend(ssim)

            mse, psnr, ssim = compute_mse_psnr_ssim(denormalize(batch_ndct), denormalize(output), is_average = False)
            print("[Denoised  out] MSE = {:.8f}, PSNR = {:.8f}, SSIM = {:.8f}".format(np.mean(mse), np.mean(psnr), np.mean(ssim)))
            mse_after.extend(mse)
            psnr_after.extend(psnr)
            ssim_after.extend(ssim)

            # test the denoising performance on real ldct
            # print("============== Test on real ldct ==============")
            dnn_pred_real = self.sess.run(self.dnn_model(self.virtual_ldct),
                            feed_dict={self.virtual_ldct: batch_ldct, self.is_training: False, K.learning_phase(): 0})
            output_real = batch_ldct - dnn_pred_real

            save_images(dnn_pred_real, [8, 8], './test/dnn_noise_real_{:04d}.png'.format(idx))
            save_images(output_real, [8, 8], './test/denoised_real_{:04d}.png'.format(idx))

            # print("1. psnr on normalized results 0~1 (batch average)")
            """
            mse, psnr, ssim = compute_mse_psnr_ssim(inverse_transform(batch_ndct), inverse_transform(batch_ldct), is_average = False)
            print("[Realdata LDCT] MSE = {:.8f}, PSNR = {:.8f}, SSIM = {:.8f}".format(np.mean(mse), np.mean(psnr), np.mean(ssim)))
            mse_real_before.extend(mse)
            psnr_real_before.extend(psnr)
            ssim_real_before.extend(ssim)

            mse, psnr, ssim = compute_mse_psnr_ssim(inverse_transform(batch_ndct), inverse_transform(output_real), is_average = False)
            print("[Denoised  out] MSE = {:.8f}, PSNR = {:.8f}, SSIM = {:.8f}".format(np.mean(mse), np.mean(psnr), np.mean(ssim)))
            mse_real_after.extend(mse)
            psnr_real_after.extend(psnr)
            ssim_real_after.extend(ssim)
            """

            # print("2. psnr on denormalized results 0~255 (batch average)")
            mse, psnr, ssim = compute_mse_psnr_ssim(denormalize(batch_ndct), denormalize(batch_ldct), is_average = False)
            print("[Realdata LDCT] MSE = {:.8f}, PSNR = {:.8f}, SSIM = {:.8f}".format(np.mean(mse), np.mean(psnr), np.mean(ssim)))
            mse_real_before.extend(mse)
            psnr_real_before.extend(psnr)
            ssim_real_before.extend(ssim)

            mse, psnr, ssim = compute_mse_psnr_ssim(denormalize(batch_ndct), denormalize(output_real), is_average = False)
            print("[Denoised  out] MSE = {:.8f}, PSNR = {:.8f}, SSIM = {:.8f}".format(np.mean(mse), np.mean(psnr), np.mean(ssim)))
            mse_real_after.extend(mse)
            psnr_real_after.extend(psnr)
            ssim_real_after.extend(ssim)
        # end of batch idx

        # The results on all test dataset
        mse_before_avg = np.mean(mse_before)
        psnr_before_avg = np.mean(psnr_before)
        ssim_before_avg = np.mean(ssim_before)
        mse_after_avg = np.mean(mse_after)
        psnr_after_avg = np.mean(psnr_after)
        ssim_after_avg = np.mean(ssim_after)

        mse_real_before_avg = np.mean(mse_real_before)
        psnr_real_before_avg = np.mean(psnr_real_before)
        ssim_real_before_avg = np.mean(ssim_real_before)
        mse_real_after_avg = np.mean(mse_real_after)
        psnr_real_after_avg = np.mean(psnr_real_after)
        ssim_real_after_avg = np.mean(ssim_real_after)

        print("============== Total results average ==============")
        print('test dataset size: ', len(data_ndct))
        print('length of mse_psnr_ssim: ', len(mse_before))

        print("============== Test on virtual ldct ==============")
        print("quantitative index on normalized results 0~255")
        print("[Virtual LDCT] MSE = {:.8f}, PSNR = {:.8f}, SSIM = {:.8f}".format(mse_before_avg, psnr_before_avg, ssim_before_avg))
        print("[Denoised out] MSE = {:.8f}, PSNR = {:.8f}, SSIM = {:.8f}".format(mse_after_avg, psnr_after_avg, ssim_after_avg))

        print("============== Test on real ldct ==============")
        print("quantitative index on normalized results 0~255")
        print("[Real LDCT] MSE = {:.8f}, PSNR = {:.8f}, SSIM = {:.8f}".format(mse_real_before_avg, psnr_real_before_avg, ssim_real_before_avg))
        print("[Denoised ] MSE = {:.8f}, PSNR = {:.8f}, SSIM = {:.8f}".format(mse_real_after_avg, psnr_real_after_avg, ssim_real_after_avg))


    def test(self, config):
        if self.load(self.checkpoint_dir):
            print(" [*] Load ckeckpoint successfully!!!")
        else:
            print(" [!] Load checkpoint failed...")

        if not os.path.exists('test'):
            os.makedirs('test')

        data_ndct = sorted(dataset_files(config.ndct))
        data_ldct = sorted(dataset_files(config.ldct))
        assert (len(data_ndct) > 0)
        assert (len(data_ldct) > 0)

        length = min(len(data_ndct), len(data_ldct))
        print('test dataset size: ', length)

        mse_real_before = []
        psnr_real_before = []
        ssim_real_before = []
        mse_real_after = []
        psnr_real_after = []
        ssim_real_after = []

        for i in range(length):
            print("============== [id = %d/%d] ==============" % (i + 1, length))

            ldct_image = [get_image(data_ldct[i], self.image_size, is_crop = self.is_crop)]
            ldct_image = np.array(ldct_image).astype(np.float32)

            ndct_image = [get_image(data_ndct[i], self.image_size, is_crop = self.is_crop)]
            ndct_image = np.array(ndct_image).astype(np.float32)

            dnn_pred_real = self.sess.run(self.dnn_model(self.virtual_ldct),
                                          feed_dict = {self.virtual_ldct: ldct_image, self.is_training: False, K.learning_phase(): 0})
            output_real = ldct_image - dnn_pred_real

            save_image(ndct_image[0], './test/clean_{:04d}.png'.format(i + 1))
            save_image(dnn_pred_real[0], './test/noise_{:04d}.png'.format(i + 1))
            save_image(output_real[0], './test/denoised_{:04d}.png'.format(i + 1))
            save_image(ldct_image[0], './test/noisy_{:04d}.png'.format(i + 1))

            mse, psnr, ssim = compute_mse_psnr_ssim(denormalize(ndct_image), denormalize(ldct_image), is_average = False)
            print("[Realdata LDCT] MSE = {:.8f}, PSNR = {:.8f}, SSIM = {:.8f}".format(np.mean(mse), np.mean(psnr), np.mean(ssim)))
            mse_real_before.extend(mse)
            psnr_real_before.extend(psnr)
            ssim_real_before.extend(ssim)

            mse, psnr, ssim = compute_mse_psnr_ssim(denormalize(ndct_image), denormalize(output_real), is_average = False)
            print("[Denoised  out] MSE = {:.8f}, PSNR = {:.8f}, SSIM = {:.8f}".format(np.mean(mse), np.mean(psnr), np.mean(ssim)))
            mse_real_after.extend(mse)
            psnr_real_after.extend(psnr)
            ssim_real_after.extend(ssim)

        mse_real_before_avg = np.mean(mse_real_before)
        psnr_real_before_avg = np.mean(psnr_real_before)
        ssim_real_before_avg = np.mean(ssim_real_before)
        mse_real_after_avg = np.mean(mse_real_after)
        psnr_real_after_avg = np.mean(psnr_real_after)
        ssim_real_after_avg = np.mean(ssim_real_after)

        print("============== Total results average ==============")
        print('test dataset size: ', length)
        print('length of mse_psnr_ssim: ', len(mse_real_before))
        print("quantitative index on normalized results 0~255: ")
        print("[Real LDCT] MSE = {:.8f}, PSNR = {:.8f}, SSIM = {:.8f}".format(mse_real_before_avg, psnr_real_before_avg, ssim_real_before_avg))
        print("[Denoised ] MSE = {:.8f}, PSNR = {:.8f}, SSIM = {:.8f}".format(mse_real_after_avg, psnr_real_after_avg, ssim_real_after_avg))
