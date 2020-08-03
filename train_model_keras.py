from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import numpy as np
import os
from scipy import misc
import tensorflow as tf
import tensorflow_addons as tfa
import keras_contrib as kc


def extract_crop(image, resolution_from, resolution_to):
    x_up = int((resolution_from[1] - resolution_to[1]) / 2)
    y_up = int((resolution_from[0] - resolution_to[0]) / 2)

    x_down = x_up + resolution_to[1]
    y_down = y_up + resolution_to[0]

    return image[y_up : y_down, x_up : x_down, :]


def conv2d(input, filters, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal'):
    return Conv2D(filters, kernel_size, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(input)

def unet(input_size = (512,512,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv1')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='mp1')(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv3')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv4')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='mp2')(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv5')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv6')(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv7')(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv8')(conv4)
    # drop4 = Dropout(0.5, name='dp1')(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv9')(conv4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv10')(conv5)
    # drop5 = Dropout(0.5, name='dp2')(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='conv11')(conv5)
    merge6 = concatenate([conv5, up6], axis=3, name='cat1')
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv12')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv13')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='conv14')(conv6)
    merge7 = concatenate([conv3, up7], axis=3, name='cat2')
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv15')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv16')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='conv17')(
        UpSampling2D(size=(2, 2), name='up1')(conv7))
    merge8 = concatenate([conv2, up8], axis=3, name='cat3')
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv18')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv19')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='conv20')(
        UpSampling2D(size=(2, 2), name='up2')(conv8))
    merge9 = concatenate([conv1, up9], axis=3, name='cat4')
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv21')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv22')(conv9)
    conv9 = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv23')(conv9)

    model = Model(inputs=inputs, outputs=conv9)

    model.compile(optimizer=Adam(lr=1e-4), loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])

    return model


def resnet_keras(input_size = (512,512,3)):
    inputs = Input(input_size)

    c1 = conv2d(inputs, 64, 9)

    # residual 1

    c2 = conv2d(c1, 64, 3)

    c3 = conv2d(c2, 64, 3)

    # residual 2

    c4 = conv2d(c3, 128, 3)

    c5 = conv2d(c4, 128, 3)

    # residual 3

    c6 = conv2d(c5, 256, 3)

    c7 = conv2d(c6, 256, 3)

    # residual 4

    c8 = conv2d(c7, 128, 3)

    c9 = conv2d(c8, 128, 3)

    # Convolutional

    c10 = conv2d(c9, 64, 3)

    c11 = conv2d(c10, 64, 3)


    # Final

    enhanced = conv2d(c11, 3, 9, activation=tf.nn.tanh)

    model = Model(inputs=inputs, outputs=enhanced)

    model.compile(optimizer=Adam(lr=1e-5), loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])

    # # model.summary()
    #
    # if (pretrained_weights):
    #     model.load_weights(pretrained_weights)

    return model


def trainGenerator(batch_size, train_path, image_folder, groundtruth_folder, image_color_mode="rgb",
                   groundtruth_color_mode="rgb", image_save_prefix="image", groundtruth_save_prefix="groundtruth",
                   save_to_dir=None, target_size=(512, 512), seed=1):
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
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[groundtruth_folder],
        class_mode=None,
        color_mode=groundtruth_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=groundtruth_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, groundtruth) in train_generator:
        img = img.astype('float32') / 255
        # img = np.reshape(img, [1,100,100,3])
        groundtruth = groundtruth.astype('float32') / 255
        # groundtruth = np.reshape(groundtruth, [1,100,100,3])
        yield (img, groundtruth)


def testGenerator(test_path,num_image = 40,target_size = (256,256),flag_multi_class = False,as_gray = False):
    for i in range(32,num_image):
        img = misc.imread(os.path.join(test_path,"%d.jpg"%i))
        # img = trans.resize(img,target_size)
        # img = (img * 255).astype('uint8')
        # img = np.reshape(img,img.shape+(3,)) if (not flag_multi_class) else img
        # img = extract_crop(img, [1356, 2048], [600, 800])
        img = np.reshape(img,[1,1080,1528,3])
        yield img


def saveResult(save_path, results, originals):
    if len(results) == 0:
        return
    for i,item in enumerate(results):
        # item = (item * 255).astype('uint8')
        misc.imsave(os.path.join(save_path,"%d_predict.jpg"%i),item)
    count = 0
    for i,item in enumerate(originals):
        misc.imsave(os.path.join(save_path,"%d_original.jpg"%i),item[0])
        count += 1
        if count == len(results):
            break



myGene = trainGenerator(1, 'dped/test/training_data', 'test', 'original', save_to_dir=None)

model = unet()
model.load_weights('keras_weights.hdf5')
# model_checkpoint = ModelCheckpoint('keras_weights.hdf5', monitor='loss', verbose=1, save_best_only=True)
# model.fit_generator(myGene, steps_per_epoch=300, epochs=5, callbacks=[model_checkpoint])
testGene = testGenerator("dped/test/test_data/full_size_test_images")
results = model.predict_generator(testGene, 1, verbose=1)
testGene = testGenerator("dped/test/test_data/full_size_test_images")
saveResult("results", results, testGene)
