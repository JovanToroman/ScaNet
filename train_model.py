# python train_model.py model={iphone,sony,blackberry} dped_dir=dped vgg_dir=vgg_pretrained/imagenet-vgg-verydeep-19.mat

import tensorflow.compat.v1 as tf
import tensorflow as tf2
from scipy import misc
import numpy as np
import sys

from load_dataset import load_test_data, load_batch
from ssim import MultiScaleSSIM
import models
import utils
import vgg

np.seterr('raise')

# defining size of the training image patches

PATCH_WIDTH = 100
PATCH_HEIGHT = 100
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3

# processing command arguments

phone, batch_size, train_size, learning_rate, num_train_iters, \
w_content, w_color, w_texture, w_tv, \
dped_dir, vgg_dir, eval_step = utils.process_command_args(sys.argv)

np.random.seed(0)

# loading training and test data

print("Loading test data...")
test_data, test_answ = load_test_data(phone, dped_dir, PATCH_SIZE, 200)
print("Test data was loaded\n")

print("Loading training data...")
train_data, train_answ = load_batch(phone, dped_dir, train_size, PATCH_SIZE)
print("Training data was loaded\n")

TEST_SIZE = test_data.shape[0]
num_test_batches = int(test_data.shape[0]/batch_size)

# defining system architecture

with tf.Graph().as_default(), tf.Session() as sess:
    
    # placeholders for training data

    phone_ = tf.placeholder(tf.float32, [None, PATCH_SIZE])
    # phone_image = tf.image.adjust_contrast(tf.image.per_image_standardization(tf.reshape(phone_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])), 0.5)
    phone_image = tf.reshape(phone_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])
    phone_image2 = tf.reshape(phone_, [-1, 256, 256, 3])

    dslr_ = tf.placeholder(tf.float32, [None, PATCH_SIZE])
    dslr_image = tf.reshape(dslr_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])


    # get processed enhanced image

    enhanced = models.unet(tf.image.rgb_to_grayscale(phone_image))

    # enhanced_gray = tf.image.rgb_to_grayscale(enhanced)
    # dslr_gray = tf.image.rgb_to_grayscale(dslr_image)
    # enhanced_gray2 = tf.reshape(tf.image.rgb_to_grayscale(enhanced), [-1, PATCH_WIDTH * PATCH_HEIGHT])
    # dslr_gray2 = tf.reshape(tf.image.rgb_to_grayscale(dslr_image), [-1, PATCH_WIDTH * PATCH_HEIGHT])


    ssim = tf.abs(tf.reduce_sum(tf.image.ssim(dslr_image, enhanced, 1.0)))
    # pixel_difference = tf.abs(tf.reduce_mean(tf.reshape(dslr_gray2, [-1])) - tf.reduce_mean(tf.reshape(enhanced_gray2, [-1])))
    #l2_loss = tf2.nn.l2_loss(dslr_image - enhanced) #not used (gives green outputs)
    # psnr = tf.abs(tf2.image.psnr(enhanced_gray, dslr_gray, max_val=1.0))

    # final loss

    loss_generator = 1/ssim

    generator_vars = [v for v in tf.global_variables() if v.name.startswith("generator")]
    train_step_gen = tf.train.AdamOptimizer(learning_rate).minimize(loss_generator, var_list=generator_vars)

    saver = tf.train.Saver()

    print('Initializing variables')
    sess.run(tf.global_variables_initializer())

    print('Training network')

    all_zeros = np.reshape(np.zeros((batch_size, 1)), [batch_size, 1])
    test_crops = test_data[np.random.randint(0, TEST_SIZE, 5), :]

    logs = open('models/' + phone + '.txt', "w+")
    logs.close()

    for i in range(num_train_iters):

        # train generator

        idx_train = np.random.randint(0, train_size, batch_size)

        phone_images = train_data[idx_train]
        dslr_images = train_answ[idx_train]

        [loss_temp, temp] = sess.run([loss_generator, train_step_gen],
                                        feed_dict={phone_: phone_images, dslr_: dslr_images})

        # train discriminator

        idx_train = np.random.randint(0, train_size, batch_size)

        # generate image swaps (dslr or enhanced) for discriminator
        swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])

        size_originals = len(train_answ)
        phone_images = train_data[idx_train]
        dslr_images = train_answ[idx_train]

        if i != 0 and i % eval_step == 0:

            # save visual results for several test image crops

            enhanced_crops = sess.run(enhanced, feed_dict={phone_: test_crops, dslr_: dslr_images})

            idx = 0
            for crop in enhanced_crops:
                before_after = np.hstack((np.reshape(test_crops[idx], [PATCH_HEIGHT, PATCH_WIDTH, 3]), crop))
                misc.imsave('results/' + str(phone)+ "_" + str(idx) + '_iteration_' + str(i) + '.jpg', before_after)
                idx += 1

            # save the model that corresponds to the current iteration

            saver.save(sess, 'models/' + str(phone) + '_iteration_' + str(i) + '.ckpt', write_meta_graph=False)

            # reload a different batch of training data

            # del train_data
            # del train_answ
            # train_data, train_answ = load_batch(phone, dped_dir, train_size, PATCH_SIZE)
        print("finised iteration no. {}".format(i))
