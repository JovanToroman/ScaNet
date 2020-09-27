from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import numpy as np
import os
# from scipy import misc
import imageio as misc
import tensorflow as tf
import tensorflow_addons as tfa
import keras_contrib as kc
from PIL import Image
from matplotlib import pyplot

from utils import concat_images


def extract_crop(image, resolution_from, resolution_to):
    x_up = int((resolution_from[1] - resolution_to[1]) / 2)
    y_up = int((resolution_from[0] - resolution_to[0]) / 2)

    x_down = x_up + resolution_to[1]
    y_down = y_up + resolution_to[0]

    return image[y_up : y_down, x_up : x_down, :]


def conv2d(input, filters, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal'):
    return Conv2D(filters, kernel_size, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(input)


def mse_plus_dssim_loss(y_true, y_pred):
    dssim = kc.losses.DSSIMObjective()(y_true, y_pred)
    # l2 = tf.reduce_sum(y_true - y_pred)
    mse = tf.keras.losses.MeanSquaredError()(tfa.image.gaussian_filter2d(y_true), tfa.image.gaussian_filter2d(y_pred))
    return dssim + 15 * mse


def unet(input_size = (128,128,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.50)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.50)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(3, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(learning_rate=1e-4), loss=mse_plus_dssim_loss, metrics=['accuracy'])

    return model



def resnet_keras(input_size = (100,100,3)):
    inputs = Input(input_size)

    c1 = conv2d(inputs, 64, 3)

    # residual 1

    c2 = conv2d(c1, 64, 3)

    c3 = concatenate([conv2d(c2, 64, 3), c1], axis=3)

    # residual 2

    c4 = conv2d(c3, 64, 3)

    c5 = concatenate([conv2d(c4, 64, 3), c3], axis=3)

    # residual 3

    c6 = conv2d(c5, 64, 3)

    c7 = concatenate([conv2d(c6, 64, 3), c5], axis=3)

    # residual 4

    c8 = conv2d(c7, 64, 3)

    c9 = concatenate([conv2d(c8, 64, 3), c7], axis=3)

    # Convolutional

    c10 = conv2d(c9, 64, 3)

    c11 = conv2d(c10, 64, 3)


    # Final

    enhanced = conv2d(c11, 3, 1)

    model = Model(inputs=inputs, outputs=enhanced)

    model.compile(optimizer=Adam(lr=1e-4), loss=mse_plus_dssim_loss, metrics=['accuracy'])

    # # model.summary()
    #
    # if (pretrained_weights):
    #     model.load_weights(pretrained_weights)

    return model


def trainGenerator(batch_size, train_path, image_folder, groundtruth_folder, image_color_mode="rgb",
                   groundtruth_color_mode="rgb", image_save_prefix="image", groundtruth_save_prefix="groundtruth",
                   save_to_dir=None, target_size=(128, 128), seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
        shuffle=True)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[groundtruth_folder],
        class_mode=None,
        color_mode=groundtruth_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=groundtruth_save_prefix,
        seed=seed,
        shuffle=True)
    train_generator = zip(image_generator, mask_generator)
    for (img, groundtruth) in train_generator:
        img = img.astype('float32') / 255
        # img = np.reshape(img, [1,100,100,3])
        groundtruth = groundtruth.astype('float32') / 255
        # groundtruth = np.reshape(groundtruth, [1,100,100,3])
        yield (img, groundtruth)


def testGenerator(test_path,num_image = 40):
    for dir in [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))]:
        for i in os.listdir(os.path.join(test_path, dir)):
            img = misc.imread(os.path.join(test_path,dir, i))
            # img = trans.resize(img,target_size)
            # img = (img * 255).astype('uint8')
            # img = np.reshape(img,img.shape+(3,)) if (not flag_multi_class) else img
            # img = extract_crop(img, [1356, 2048], [600, 800])
            img = img.astype('float32') / 255
            img = img[np.newaxis, ...] # add first dimension 1 to nparray
            image_name = '{}_{}_{}'.format(i.split('_')[0], dir, i.split('_')[1])
            yield (img, image_name)


def originalGenerator(test_path,num_image = 40):
    for i in range(0,num_image):
        img = misc.imread(os.path.join(test_path,"%d.jpg"%i))
        # img = trans.resize(img,target_size)
        # img = (img * 255).astype('uint8')
        # img = np.reshape(img,img.shape+(3,)) if (not flag_multi_class) else img
        # img = extract_crop(img, [1356, 2048], [600, 800])
        img = img[np.newaxis, ...] # add first dimension 1 to nparray
        yield img


def saveResult(save_path, results, inputs):
    if len(results) == 0:
        return
    count = 0
    for item in results:
        image_name, img = item
        original_image_name = image_name.split('_')[0] + '.jpg'
        # item = (item * 255).astype('uint8')
        processed_image = Image.fromarray((img[0]*255).astype('uint8'))
        original_image = Image.open(os.path.join('origim', original_image_name))
        input_image = Image.fromarray((next(inputs)[0][0]*255).astype('uint8'))
        concat_images([input_image, processed_image, original_image], os.path.join(save_path, image_name))
        # misc.imsave(os.path.join(save_path, image_name),img[0])
        # misc.imsave(os.path.join(save_path, image_name), )
        count += 1


train = 1

if train == 1:
    myGene = trainGenerator(64, 'dped/test/training_data', 'test1', 'original1', save_to_dir=None)
    model = unet()
    # model.load_weights('keras_weights.hdf5')
    model_checkpoint = ModelCheckpoint(
        'konacni_trening_0.5dropout_{epoch}_{loss}_{accuracy}.hdf5', monitor='loss', verbose=1, save_best_only=True)
    history = model.fit_generator(myGene, steps_per_epoch=18122
                        , epochs=5, callbacks=[model_checkpoint])
    # x1,x2,y1,y2 = pyplot.axis()
    # pyplot.axis([0, 4, 0, 1])
    # pyplot.plot(history.history['loss'])
    # pyplot.plot(history.history['accuracy'])
    # pyplot.legend(['loss','accuracy'], loc='upper left')
    # pyplot.show()
else:
    results = []
    count = 0
    num_to_evaluate = 10
    while(num_to_evaluate == -1 or count < num_to_evaluate):
        testGene = testGenerator("dped/test/test_data/full_size_test_images", num_image=150)
        for i in range(len(results)):
            try: next(testGene)
            except: break
        try: img, image_name = next(testGene)
        except Exception as e: print("Exception: " + str(e))
        model = unet(input_size=img[0].shape)
        model.load_weights('keras_weights_radeci_model_sa_sigmoidkom_3.hdf5')
        predict_one = model.predict(img)
        results.append((image_name, predict_one))
        count += 1
        print("{} done!".format(count))
    testGene = testGenerator("dped/test/test_data/full_size_test_images", num_image=23)
    saveResult("results", results, testGene)
