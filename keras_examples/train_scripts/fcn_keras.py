## Example code 
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.initializers import Constant
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint
from PIL import Image
import numpy as np
import argparse
import copy
import os

TRAIN_DATA = "./list/train_id.txt"
IMAGE_DIR = "./data/pascal_voc12/images_orig/"
LABEL_DIR = "./data/pascal_voc12/labels_orig/"

nb_classes = 21
# Bilinear interpolation (reference: https://github.com/warmspringwinds/tf-image-segmentation/blob/master/tf_image_segmentation/utils/upsampling.py)
def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = factor*2 - factor%2
    factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:filter_size, :filter_size]
    upsample_kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
                       dtype=np.float32)
    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights

def fcn_32s():
    inputs = Input(shape=(None, None, 3))
    vgg16 = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
    x = Conv2D(filters=nb_classes, 
               kernel_size=(1, 1))(vgg16.output)
    x = Conv2DTranspose(filters=nb_classes, 
                        kernel_size=(64, 64),
                        strides=(32, 32),
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer=Constant(bilinear_upsample_weights(32, nb_classes)))(x)
    model = Model(inputs=inputs, outputs=x)
    for layer in model.layers[:15]:
        layer.trainable = False
    return model

def load_image(path):
    img_org = Image.open(path)
    w, h = img_org.size
    img = img_org.resize(((w//32)*32, (h//32)*32))
    img = np.array(img, dtype=np.float32)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    return x

def load_label(path):
    img_org = Image.open(path)
    w, h = img_org.size
    img = img_org.resize(((w//32)*32, (h//32)*32))
    img = np.array(img, dtype=np.uint8)
    img[img==255] = 0
    y = np.zeros((1, img.shape[0], img.shape[1], nb_classes), dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            y[0, i, j, img[i][j]] = 1
    return y

def generate_arrays_from_file(path, image_dir, label_dir):
    while 1:
        f = open(path)
        for line in f:
            filename = line.rstrip('\n')
            path_image = os.path.join(image_dir, filename+'.jpg')
            path_label = os.path.join(label_dir, filename+'.png')
            x = load_image(path_image)
            y = load_label(path_label)
            yield (x, y)
        f.close()

def model_predict(model, input_path, output_path):
    img_org = Image.open(input_path)
    w, h = img_org.size
    img = img_org.resize(((w//32)*32, (h//32)*32))
    img = np.array(img, dtype=np.float32)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)
    pred = pred[0].argmax(axis=-1).astype(np.uint8)
    img = Image.fromarray(pred, mode='P')
    img = img.resize((w, h))
    palette_im = Image.open("/storage/cfmata/deeplab/datasets/voc12/train/VOCdevkit/VOC2012/SegmentationClass/2008_005637.png")
    img.palette = copy.copy(palette_im.palette)
    img.save(output_path)


if __name__ == "__main__":
    nb_data = sum(1 for line in open(TRAIN_DATA))
    model = fcn_32s()
    model.compile(loss="categorical_crossentropy", optimizer='sgd')
    for epoch in range(100):
        model.fit_generator(
            generate_arrays_from_file(TRAIN_DATA, IMAGE_DIR, LABEL_DIR),
            steps_per_epoch=nb_data, 
            epochs=1)
        model_predict(model, 'test.jpg', 'predict-{}.png'.format(epoch))
    model.save_weights("fcn_keras_epoch_100.h5")
