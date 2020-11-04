# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from random import shuffle
from absl import app
from absl import flags

from modified_model import *

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import imageio
import sys


flags.DEFINE_string("A_txt_path", "D:/[2]DB/cyclegan_DB_cropsetting/trainA_half_littleface.txt", "A image txt directory")

flags.DEFINE_string("A_img_path", "D:/[2]DB/cyclegan_DB_cropsetting/trainA_half_littleface/", "A image directory")

flags.DEFINE_integer("n_A_images", 7988, "Number of A images")

flags.DEFINE_string("B_txt_path", "D:/[2]DB/cyclegan_DB_cropsetting/trainB_half_littleface.txt", "B image txt directory")

flags.DEFINE_string("B_img_path", "D:/[2]DB/cyclegan_DB_cropsetting/trainB_half_littleface/", "B image directory")

flags.DEFINE_integer("n_B_images", 13215, "Number of B images")

flags.DEFINE_bool("pre_checkpoint", True, "Fine tune or continue the train")

flags.DEFINE_string("pre_checkpoint_path", "C:/Users/Yuhwan/Desktop/1249", "Fine tuning or continue the train")

flags.DEFINE_string("save_checkpoint", "D:/tensorflow2.0(CycleGAN)/checkpoint", "Save checkpoint")

flags.DEFINE_integer("epochs", 200, "Training epoch")

flags.DEFINE_integer('epoch_decay', 100, 'Epoch decay')

#flags.DEFINE_integer("options", 64, "Defalut filter size")

flags.DEFINE_integer("batch_size", 1, "Traing batch")

flags.DEFINE_integer("img_size", 256, "Image size")

flags.DEFINE_integer("ch", 3, "Image channel")

flags.DEFINE_integer("load_size", 286, "Before cropped image")

flags.DEFINE_float('lr', 2e-4, 'Learning rate')

flags.DEFINE_bool('train', False, 'True or False')

flags.DEFINE_string('sample_dir', 'D:/tensorflow2.0(CycleGAN)', 'sample path')

#############################################################################################################################
flags.DEFINE_string('A_test_img', 'D:/[1]DB/[1]second_paper_DB/[1]First_fold/_MORPH_MegaAge_16_43_fullDB/[1]FullDB/testA/', 'A test image path')

flags.DEFINE_string('A_test_txt', 'D:/[1]DB/[1]second_paper_DB/[1]First_fold/_MORPH_MegaAge_16_43_fullDB/[3]MegaAge_30_43_and_Morph_16_29/MegaAge_test_30_43.txt', 'A text path')

#flags.DEFINE_integer('A_n_images', 5279, 'Number of A images')

flags.DEFINE_string('A_test_output', 'D:/[1]DB/[1]second_paper_DB/[1]First_fold/modified_CycleGAB_fullDB/_MegaAge_Morph_16_43_fullDB/[3_2]Trans_MegaAge_16_29', 'path of generating A images')

flags.DEFINE_string('B_test_img', 'D:/[1]DB/[1]second_paper_DB/[1]First_fold/_MORPH_MegaAge_16_43_fullDB/[1]FullDB/testB/', 'B test image path')

flags.DEFINE_string('B_test_txt', 'D:/[1]DB/[1]second_paper_DB/[1]First_fold/_MORPH_MegaAge_16_43_fullDB/[3]MegaAge_30_43_and_Morph_16_29/Morph_test_16_29.txt', 'B text path')

#flags.DEFINE_integer('B_n_images', 5279, 'Number of B images')

flags.DEFINE_string('B_test_output', 'D:/[1]DB/[1]second_paper_DB/[1]First_fold/modified_CycleGAB_fullDB/_MegaAge_Morph_16_43_fullDB/[3_3]Trans_Morph_30_43', 'path of generating B images')

flags.DEFINE_string('test_dir', 'B2A', 'A2B or B2A')
#############################################################################################################################

FLAGS = flags.FLAGS
FLAGS(sys.argv)

len_dataset = min(FLAGS.n_A_images, FLAGS.n_B_images)
G_lr_scheduler = LinearDecay(FLAGS.lr, FLAGS.epochs * len_dataset, FLAGS.epoch_decay * len_dataset)
D_lr_scheduler = LinearDecay(FLAGS.lr, FLAGS.epochs * len_dataset, FLAGS.epoch_decay * len_dataset)
generator_optimizer = tf.keras.optimizers.Adam(G_lr_scheduler, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(D_lr_scheduler, beta_1=0.5)

def abs_criterion(input, target):
    return tf.reduce_mean(tf.abs(input - target))

def mae_criterion(input, target):
    return tf.reduce_mean((input - target)**2)

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels, logits))

@tf.function        # ???????? ???? ?? ?κ??? ???????? ?Ѵ?.
def train(A_images, B_images, generator_A2B, generator_B2A, discriminator_A, discriminator_B):
    with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
        fake_B, fake_Blogits = generator_A2B(A_images, training=True)
        fake_A_, fake_A_logits = generator_B2A(fake_B, training=True)              
        fake_A, fake_Alogits = generator_B2A(B_images, training=True)       
        fake_B_, fake_B_logits = generator_A2B(fake_A, training=True)              
                        
        DB_fake, DB_fake_logit = discriminator_A(fake_A, training=True)            
        DA_fake, DA_fake_logit = discriminator_B(fake_B, training=True)            
        DA_real, DA_real_logit = discriminator_A(A_images, training=True)
        DB_real, DB_real_logit = discriminator_B(B_images, training=True)

        gen_tape.watch(fake_A)
        gen_tape.watch(fake_B)
        gen_tape.watch(fake_A_)
        gen_tape.watch(fake_B_)

        disc_tape.watch(DB_fake)
        disc_tape.watch(DB_real)
        disc_tape.watch(DA_fake)
        disc_tape.watch(DA_real)

        AB_energyft_g = tf.reduce_mean(tf.math.abs(fake_Blogits[0:54] - fake_A_logits[0:54]))
        L_AB_energyft_g = 2 * 100 * tf.math.exp(AB_energyft_g * (-2.77)/100)

        BA_enregyft_g = tf.reduce_mean(tf.math.abs(fake_Alogits[0:54] - fake_B_logits[0:54]))    
        L_BA_enregyft_g = 2 * 100 * tf.math.exp(BA_enregyft_g * (-2.77)/100) 

        AB_energyft = tf.reduce_mean(tf.math.abs(DA_real_logit[0:54] - DB_fake_logit[0:54]))      
        L_AB_energyft = AB_energyft*AB_energyft*2/100

        BA_energyft = tf.reduce_mean(tf.math.abs(DB_real_logit[0:54] - DA_fake_logit[0:54]))
        L_BA_energyft = BA_energyft*BA_energyft*2/100

        g_a2b_loss = mae_criterion(DB_fake, tf.ones_like(DB_fake)) + (10.0 * abs_criterion(A_images, fake_A_)) + (10.0 * abs_criterion(B_images, fake_B_))
        g_b2a_loss = mae_criterion(DA_fake, tf.ones_like(DA_fake)) + (10.0 * abs_criterion(A_images, fake_A_)) + (10.0 * abs_criterion(B_images, fake_B_))
        disc_A_loss = (mae_criterion(DA_real, tf.ones_like(DA_real)) + mae_criterion(DA_fake, tf.zeros_like(DA_fake))) / 2
        disc_B_loss = (mae_criterion(DB_real, tf.ones_like(DB_real)) + mae_criterion(DB_fake, tf.zeros_like(DB_fake))) / 2
                
        g_loss = mae_criterion(DB_fake, tf.ones_like(DB_fake)) + mae_criterion(DA_fake, tf.ones_like(DA_fake)) \
                + (10.0 * abs_criterion(A_images, fake_A_)) + (10.0 * abs_criterion(B_images, fake_B_)) + L_AB_energyft_g + L_BA_enregyft_g
    
        d_loss = disc_A_loss + disc_B_loss + L_AB_energyft + L_BA_energyft

    generator_gradients = gen_tape.gradient(g_loss, generator_A2B.trainable_variables + generator_B2A.trainable_variables)
    discriminator_gradients = disc_tape.gradient(d_loss, discriminator_A.trainable_variables + discriminator_B.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator_A2B.trainable_variables + generator_B2A.trainable_variables))
    
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator_A.trainable_variables + discriminator_B.trainable_variables))

    return g_loss, d_loss
######################################################################################################################################

def _func(A_filename, B_filename):

    h = tf.random.uniform([1], 1e-2, 30)
    h = tf.cast(tf.math.ceil(h[0]), tf.int32)
    w = tf.random.uniform([1], 1e-2, 30)
    w = tf.cast(tf.math.ceil(w[0]), tf.int32)

    A_image_string = tf.io.read_file(A_filename)
    A_image_decode = tf.image.decode_jpeg(A_image_string, channels=3)
    A_image_decode = tf.image.resize(A_image_decode, [FLAGS.img_size + 30, FLAGS.img_size + 30])
    A_image_decode = A_image_decode[h:h+FLAGS.img_size, w:w+FLAGS.img_size, :]
    A_image_decode = tf.image.convert_image_dtype(A_image_decode, tf.float32) / 127.5 - 1.

    B_image_string = tf.io.read_file(B_filename)
    B_image_decode = tf.image.decode_jpeg(B_image_string, channels=3)
    B_image_decode = tf.image.resize(B_image_decode, [FLAGS.img_size + 30, FLAGS.img_size + 30])
    B_image_decode = B_image_decode[h:h+FLAGS.img_size, w:w+FLAGS.img_size, :]
    B_image_decode = tf.image.convert_image_dtype(B_image_decode, tf.float32) / 127.5 - 1.

    if tf.random.uniform(()) > 0.5:
        A_image_decoded = tf.image.flip_left_right(A_image_decode)
        B_image_decoded = tf.image.flip_left_right(B_image_decode)

    return A_image_decoded, B_image_decoded

def _test_func(file):

    A_image_string = tf.io.read_file(file)
    A_image_decode = tf.image.decode_jpeg(A_image_string, channels=3)
    A_image_decode = tf.image.resize(A_image_decode, [FLAGS.img_size, FLAGS.img_size])
    A_image_decode = tf.image.convert_image_dtype(A_image_decode, tf.float32) / 127.5 - 1.

    return A_image_decode

def main(argv=None):
    
    discriminator_A = ConvDiscriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.ch))
    discriminator_B = ConvDiscriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.ch))
    generator_A2B = ResnetGenerator(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.ch))
    generator_B2A = ResnetGenerator(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.ch))
   
    if FLAGS.pre_checkpoint is True:
    
        ckpt = tf.train.Checkpoint(generator_A2B=generator_A2B,
                           generator_B2A=generator_B2A,
                           discriminator_A=discriminator_A,
                           discriminator_B=discriminator_B,
                           generator_optimizer=generator_optimizer,
                           discriminator_optimizer=discriminator_optimizer)

        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')

    if FLAGS.train == True:

        Adata_name = np.loadtxt(FLAGS.A_txt_path, dtype='<U100', skiprows=0, usecols=0)
        Adata_name = [FLAGS.A_img_path + Adata_name_ for Adata_name_ in Adata_name]

        Bdata_name = np.loadtxt(FLAGS.B_txt_path, dtype='<U100', skiprows=0, usecols=0)
        Bdata_name = [FLAGS.B_img_path + Bdata_name_ for Bdata_name_ in Bdata_name]
        
        count = 0
        batch_idxs = min(min(len(Adata_name), len(Bdata_name)), 50000) // FLAGS.batch_size
        for i in range(FLAGS.epochs):
            Adata_name = shuffle(Adata_name)
            Bdata_name = shuffle(Bdata_name)
            data_generator = tf.data.Dataset.from_tensor_slices((Adata_name, Bdata_name))
            data_generator = data_generator.shuffle(min(len(Adata_name), len(Bdata_name)))
            data_generator = data_generator.map(_func)
            data_generator = data_generator.batch(FLAGS.batch_size)
            data_generator = data_generator.prefetch(tf.data.experimental.AUTOTUNE)

            it = iter(data_generator)
            for step in range(batch_idxs):
    
                A_batch_images, B_batch_images = next(it)
    
                ###########################################################################################
                g_loss, d_loss = train(A_batch_images, B_batch_images, generator_A2B, generator_B2A, discriminator_A, discriminator_B)
                
                print(("Epoch: [{}] [{}/{}], G_loss:{}, D_loss:{}".format(i + 1, step, batch_idxs, g_loss, d_loss)))
                print(discriminator_optimizer.iterations.numpy())
    
                if count % 500 == 0:
                    fake_B_image, _ = generator_A2B(A_batch_images, training=False)
                    fake_A_image, _ = generator_B2A(B_batch_images, training=False)
                    
                    plt.imsave("{}/fake_A_{}steps.jpg".format(FLAGS.sample_dir,count), fake_A_image[0] * 0.5 + 0.5)
                    plt.imsave("{}/fake_B_{}steps.jpg".format(FLAGS.sample_dir,count), fake_B_image[0] * 0.5 + 0.5)
                    plt.imsave("{}/real_A_{}steps.jpg".format(FLAGS.sample_dir,count), A_batch_images[0] * 0.5 + 0.5)
                    plt.imsave("{}/real_B_{}steps.jpg".format(FLAGS.sample_dir,count), B_batch_images[0] * 0.5 + 0.5)
               
                if count % 1000 == 0:
                    model_dir = FLAGS.save_checkpoint
                    folder_name = int(count/1000)
                    folder_neme_str = '%s/%s' % (model_dir, folder_name)
                    if not os.path.isdir(folder_neme_str):
                        print("Make {} folder to save checkpoint".format(folder_name))
                        os.makedirs(folder_neme_str)
                    checkpoint = tf.train.Checkpoint(generator_A2B=generator_A2B,
                                                    generator_B2A=generator_B2A,
                                                    discriminator_A=discriminator_A,
                                                    discriminator_B=discriminator_B,
                                                    generator_optimizer=generator_optimizer,
                                                    discriminator_optimizer=discriminator_optimizer)
                    checkpoint_dir = folder_neme_str + "/" + "CycleGAN_model_{}_steps.ckpt".format(count + 1)
                    checkpoint.save(checkpoint_dir)
    
    
                count += 1
                ###########################################################################################
    else:
        if FLAGS.test_dir == 'A2B':
            Adata_name = np.loadtxt(FLAGS.A_test_txt, dtype='<U100', skiprows=0, usecols=0)
            Adata_name = [FLAGS.A_test_img + data for data in Adata_name]
            
            data_generator = tf.data.Dataset.from_tensor_slices(Adata_name)
            data_generator = data_generator.map(_test_func)
            data_generator = data_generator.batch(1)
            data_generator = data_generator.prefetch(tf.data.experimental.AUTOTUNE)
            print('==================')
            print('Start to A2B......')
            print('==================')
            it = iter(data_generator)
            for i in range(len(Adata_name)):

                image = next(it)
                filename = Adata_name[i].split('/')[7]
                fake_B, _ = generator_A2B(image, training=False)
                plt.imsave(u'{}/{}'.format(FLAGS.A_test_output, filename), fake_B[0]*0.5+0.5)
                if i % 1000 == 0:
                    print('Generated {} image(s)..'.format(i + 1))

        else:
            Bdata_name = np.loadtxt(FLAGS.B_test_txt, dtype='<U100', skiprows=0, usecols=0)
            Bdata_name = [FLAGS.B_test_img + data for data in Bdata_name]

            data_generator = tf.data.Dataset.from_tensor_slices(Bdata_name)
            data_generator = data_generator.map(_test_func)
            data_generator = data_generator.batch(1)
            data_generator = data_generator.prefetch(tf.data.experimental.AUTOTUNE)
            print('==================')
            print('Start to B2A......')
            print('==================')
            it = iter(data_generator)
            for i in range(len(Bdata_name)):

                image = next(it)            
                filename = Bdata_name[i].split('/')[7]
                fake_A, _ = generator_B2A(image, training=False)
                plt.imsave(u'{}/{}'.format(FLAGS.B_test_output, filename), fake_A[0]*0.5+0.5)
                if i % 1000 == 0:
                    print('Generated {} image(s)..'.format(i + 1))            




if __name__ == '__main__':
     app.run(main)