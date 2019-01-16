# eval:

import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.layers import *
import pandas as pd
import argparse
import ntpath
#
import numpy as np
from models import load_model_gby
from datasets import load_dataset, load_testset
import matplotlib.pyplot as plt
from utils import IoU_ver2,give_color_to_seg_img, model_predict_gby, load_segmentations
import pdb
import scipy.misc
from PIL import Image
import matplotlib

def argument_parser_eval():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('-m', '--model', default='fcn_RESNET50_8s', help='choose between \'fcn_VGG16_32s\',\'fcn_VGG16_8s\',\'fcn_RESNET50_32s\', and \'fcn_RESNET50_8s\' networks, with or without \'_crfrnn\' suffix', type=str)
    parser.add_argument('-w', '--weights', default=None, nargs='?', const=None, help='The absolute path of the weights',type=str)
    parser.add_argument('-ds', '--dataset', default='streets', help='The name of train/test sets', type=str)
    parser.add_argument('-vb', '--verbosemode', default=1, help='Specify the verbose mode',type=int)
    parser.add_argument('-sp', '--superpixel', default=0, help='Use the superpixel cliques or not', type=int)
    return parser.parse_args()

# ===========================
# Main
# ===========================
if __name__ == '__main__':

    # ===============
    # INTRO
    # ===============

    # Parse args:
    # -----------
    args = argument_parser_eval()

    # Import Keras and Tensorflow to develop deep learning FCN models:
    # -----------------------------------------------------------------
    warnings.filterwarnings("ignore")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.70 #0.95
    config.gpu_options.visible_device_list = "1"
    set_session(tf.Session(config=config))

    print("python {}".format(sys.version))
    print("keras version {}".format(keras.__version__)) #; del keras
    print("tensorflow version {}".format(tf.__version__))

    # -----------------------------------------
    # ===============
    # LOAD train data:
    # ===============

    INPUT_SIZE = 512  # #500 #224 #512 # NOTE: Extract from model
    
    #seg_train, seg_test = [], []
    #if args.superpixel == 1:
    #    seg_train, seg_test = load_segmentations(args.dataset, INPUT_SIZE)
        
    ds = load_dataset(args.dataset, INPUT_SIZE)
    print(ds.X_train.shape, ds.y_train.shape)
    print(ds.X_test.shape, ds.y_test.shape)
    nb_classes = ds.nb_classes

    num_crf_iterations = 10  # at test time

    # ===============
    # LOAD model:
    # ===============

    model_name = args.model
    model_path_name = args.weights

    print('====================================================================================')
    print(model_path_name)
    print('====================================================================================')

    model = load_model_gby(model_name, INPUT_SIZE, nb_classes, num_crf_iterations)

    #loading weights:
    model.load_weights(model_path_name)

    batchsize = 32
    if model.crf_flag:
        batchsize = 1

    # Predictions
    #for img in ds.X_test:
    #    model_predict_gby(model, "data/horse_fine_parts/images_orig/"+img)

    # ===============
    # ANALYZE model:
    # ===============
    
    # Compute IOU:
    # ------------
    print('computing mean IoU for validation set..')
    y_pred = model.predict(ds.X_test,batch_size=batchsize, verbose=1) #[ds.X_test, seg_test]
        
    y_predi = np.argmax(y_pred, axis=3)
    y_testi = np.argmax(ds.y_test, axis=3)
    print(y_testi.shape, y_predi.shape)
    IoU_ver2(y_testi, y_predi)
    '''
    # Generate image predictions
    for i in range(len(y_pred)):
        seg = y_pred[i]
        print(seg.shape)
        matplotlib.image.imsave("image_results/horse_fine/fcn/"+str(i)+".png", seg)
        #im = Image.fromarray(seg)
        #im.save("image_results/horse_fine/fcn/"+str(i)+".png")
    '''
    #    color_seg = give_color_to_seg_img(seg, nb_classes)
    #    scipy.misc.imsave("image_results/horse_fine/fcn_crf/"+ str(i)+ ".jpg", color_seg)
        #util.get_label_image(seg, img_h, img_w)
        #segmentation.save("image_results/horse_fine/fcn/"+str(i)+".png")
    
    # Visualize the model performance:
    # --------------------------------
    shape = (INPUT_SIZE, INPUT_SIZE)
    n_classes = nb_classes # 10

    num_examples_to_plot = 4
    for i in range(len(y_predi)):
        fig = plt.figure()
        seg = y_predi[i]
        ax = fig.add_subplot(111)
        ax.imshow(give_color_to_seg_img(seg, n_classes))
        plt.savefig("image_results/pascal_voc/fcn_pred/"+ str(i)+ ".jpg")
        fig = plt.figure()
        segtest = y_testi[i]
        ax = fig.add_subplot(111)
        ax.imshow(give_color_to_seg_img(segtest, n_classes))
        plt.savefig("image_results/pascal_voc/fcn_gt/"+ str(i)+ ".jpg")

    keras.backend.clear_session()
'''
    fig = plt.figure(figsize=(10, 3*num_examples_to_plot))
    fig2 = plt.figure()
    for i in range(num_examples_to_plot):
        #img_indx = i
        img_indx = i*4
        img_is = (ds.X_test[img_indx] + 1) * (255.0 / 2)
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
        
        ax1 = fig2.add_subplot()
        ax1.imshow(give_color_to_seg_img(segtest, n_classes))
        plt2.savefig("image_results/horse_fine/fcn_crf/"+ str(i)+ ".jpg")
        
        if i == 0:
            ax.set_title("true class")

    plt.savefig("image_results/horse_fine/fcn_crf/"+ str(i)+ ".jpg")

    # clear model: Destroys the current TF graph and creates a new one. Useful to avoid clutter from old models / layers.
    keras.backend.clear_session()
'''
# usage:
# >>python eval_gby.py
