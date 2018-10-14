"""
MIT License

"""
import numpy as np
import matplotlib.pyplot as plt
import pdb
from models_gby import fcn_8s_Sadeep, fcn_8s_Sadeep_crfrnn
from utils_gby import generate_arrays_from_file,extract_arrays_from_file,IoU,model_predict_gby,getImageArr,getSegmentationArr,IoU_ver2,give_color_to_seg_img,visualize_conv_filters
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
import pandas as pd
from sklearn.utils import shuffle
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import argparse
import pickle
#from tensorflow.python import debug as tf_debug

## location of VGG weights
VGG_Weights_path = "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

RES_DIR = "/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/results/streets/"

INPUT_SIZE = 500 #224

# ===========================
# Main
# ===========================
if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.visible_device_list = "0"
    sess = tf.InteractiveSession(config=config)
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    set_session(sess)

    # Data processing: (for streets dataset)
    nb_classes = 12
    dir_img = '/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/data/streets/images_prepped_train/'
    dir_seg = '/storage/cfmata/deeplab/crf_rnn/crfasrnn_keras/data/streets/annotations_prepped_train/'
    images = sorted(os.listdir(dir_img))
    segmentations = sorted(os.listdir(dir_seg))
    
    X = []
    Y = []
    for im, seg in zip(images, segmentations):
        X.append(getImageArr(dir_img + im, INPUT_SIZE, INPUT_SIZE))
        Y.append(getSegmentationArr(dir_seg + seg, INPUT_SIZE, INPUT_SIZE, nb_classes))
    X, Y = np.array(X), np.array(Y)
    print("Images and Segmentations arrays", X.shape, Y.shape)

    # Split between training and testing data:
    train_rate = 0.85
    allow_randomness = False

    if allow_randomness:
        index_train = np.random.choice(X.shape[0], int(X.shape[0] * train_rate), replace=False)
        index_test = list(set(range(X.shape[0])) - set(index_train))
        X, Y = shuffle(X, Y)
        X_train, y_train = X[index_train], Y[index_train]
        X_test, y_test = X[index_test], Y[index_test]
    else:
        index_train = int(X.shape[0] * train_rate)
        X_train, y_train = X[0:index_train], Y[0:index_train]
        X_test, y_test = X[index_train:-1], Y[index_train:-1]

    X_train, y_train = X_train[0:1], y_train[0:1]
    X_test, y_test = X_test[0:1], y_test[0:1]
    print("train shape ", X_train.shape, y_train.shape)
    print("test shape ", X_test.shape, y_test.shape)


    # Constructing model:
    #model = fcn_8s_Sadeep(nb_classes)
    model = fcn_8s_Sadeep_crfrnn(nb_classes)

    # if resuming training:
    #saved_model_path = '/storage/gby/semseg/streets_weights_fcn8s_Sadeep_500ep' #'crfrnn_keras_model.h5'
    #saved_model_path = '/storage/gby/semseg/voc12_weights'
    #saved_model_path = 'crfrnn_keras_model.h5'
    #saved_model_path = './checkpoint/streets/weights.01-6.08'
    #model.load_weights(saved_model_path)

    model.summary()
    
    #layer visualization
    #de/conv layers: 2,3, 5,6, 8,9,10, 12,13,14, 16,17,18, 20,22,23,24, 27
    '''
    for i in range(len(model.layers)):
        if model.layers[i].name == 'conv1_2':
            print("conv1_2 weights")
            print(model.layers[i].get_weights())
        if model.layers[i].name == 'score2':
            print("score2 weights")
            print(model.layers[i].get_weights())
    visualize_conv_filters(model, INPUT_SIZE, 'conv1_2')
    visualize_conv_filters(model, INPUT_SIZE, 'score2')
    '''
    # Training starts here:
    sgd = optimizers.SGD(lr=1e-13, momentum=0.99) # Can try for the crf layer
    #sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False) # Good for the fcn_8s_Sadeep model (ie, no crf)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # for crfrnn:
    checkpointer = ModelCheckpoint(filepath='./checkpoint/streets/' + 'starting_1_weights.{epoch:02d}-{val_loss:.2f}', verbose=1, save_best_only=False, save_weights_only=True, period=1)
    hist1 = model.fit(X_train, y_train,
                      validation_data=(X_test, y_test),
                      batch_size=1, epochs=100, verbose=2, callbacks=[checkpointer])

    # Add debugging tools
    #hooks = [tf_debug.LocalCLIDebugHook()]

    #Test model on random input
    #x = np.ones(dtype="float32", shape=(1,500,500,3))
    #output = model.predict([x], batch_size=1, verbose=0, steps=None)
    
    # save model:
    model.save_weights(RES_DIR + 'streets_weights')

    save_graphics_mode = False
    print_IoU_flag = True

    # Plot/save the change in loss over epochs:
    # -------------------------------------
    '''
    with open('trainHistoryDict', 'wb') as file_pi:
        pickle.dump(hist1.history, file_pi)
    
    if(save_graphics_mode):
        for key in ['loss', 'val_loss']:
            plt.plot(hist1.history[key], label=key)
        plt.legend()
        #plt.show(block=False)
        plt.savefig('loss_plot.pdf')

    # Compute IOU:
    # ------------
    if(print_IoU_flag):
        print('computing mean IoU for validation set..')
        y_pred = model.predict(X_test)
        y_predi = np.argmax(y_pred, axis=3)
        y_testi = np.argmax(y_test, axis=3)
        print(y_testi.shape, y_predi.shape)
        IoU_ver2(y_testi, y_predi)

    # Visualize the model performance:
    # --------------------------------
    shape = (INPUT_SIZE, INPUT_SIZE)
    n_classes = nb_classes # 10

    if save_graphics_mode:

        num_examples_to_plot = 4

        fig = plt.figure(figsize=(10, 3*num_examples_to_plot))

        for i in range(num_examples_to_plot):

            img_indx = i*4
            img_is = (X_test[img_indx] + 1) * (255.0 / 2)
            seg = y_predi[img_indx]
            segtest = y_testi[img_indx]

            ax = fig.add_subplot(num_examples_to_plot, 3, 3 * i + 1)
            ax.imshow(img_is / 255.0)
            if i == 0:
                ax.set_title("original")

            ax = fig.add_subplot(num_examples_to_plot, 3, 3 * i + 2)
            ax.imshow(give_color_to_seg_img(seg, n_classes))
            if i == 0:
                ax.set_title("predicted class")

            ax = fig.add_subplot(num_examples_to_plot, 3, 3 * i + 3)
            ax.imshow(give_color_to_seg_img(segtest, n_classes))
            if i == 0:
                ax.set_title("true class")

        plt.savefig('examples.png')

    # Predict 1 test exmplae and save:
    #model_predict_gby(model, 'image.jpg', 'predict-final.png', INPUT_SIZE)
    '''
