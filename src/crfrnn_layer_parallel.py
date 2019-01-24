"""
MIT License
Copyright (c) 2017 Sadeep Jayasumana
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer
import high_dim_filter_loader
custom_module = high_dim_filter_loader.custom_module
import pdb

def _diagonal_initializer(shape):
    return np.eye(shape[0], shape[1], dtype=np.float32)


def _potts_model_initializer(shape):
    return -1 * _diagonal_initializer(shape)

def _batch_init(shape):
    return 0

class CrfRnnLayer(Layer):
    """ Implements the CRF-RNN layer described in:
    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015
    """

    def __init__(self, image_dims, num_classes,
                 theta_alpha, theta_beta, theta_gamma, batch_size, batch_sizes_train, batch_sizes_val, batch_sizes_total,
                 num_iterations, **kwargs):
        self.image_dims = image_dims
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        
        self.batch_size = batch_size
        self.batch_sizes_train = batch_sizes_train
        self.batch_sizes_val = batch_sizes_val
        self.total_batches = batch_sizes_total
        #self.state = 0
        
        self.num_iterations = num_iterations
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None
        super(CrfRnnLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Weights of the spatial kernel
        self.spatial_ker_weights = self.add_weight(name='spatial_ker_weights',
                                                   shape=(self.num_classes, self.num_classes),
                                                   initializer=_diagonal_initializer,
                                                   trainable=True)

        # Weights of the bilateral kernel
        self.bilateral_ker_weights = self.add_weight(name='bilateral_ker_weights',
                                                     shape=(self.num_classes, self.num_classes),
                                                     initializer=_diagonal_initializer,
                                                     trainable=True)

        # Compatibility matrix
        self.compatibility_matrix = self.add_weight(name='compatibility_matrix',
                                                    shape=(self.num_classes, self.num_classes),
                                                    initializer=_potts_model_initializer,
                                                    trainable=True)

        '''
        self.state = self.add_weight(name = 'batch',
                                     shape=(1),
                                     initializer=_batch_init,
                                     trainable=False)
        '''
        super(CrfRnnLayer, self).build(input_shape)

    def call(self, inputs):
        # python lists for variables
        unary_list, rgb_list, q_values_list, = [], [], []

        #if inputs[0].get_shape().as_list()[0] == None:
        #    num_imgs = 1
        #else:
        #    num_imgs = inputs[0].get_shape().as_list()[0] # gets number of images in current batch
        #num_imgs = tf.placeholder(tf.int32, shape=(1))
        #ones = tf.ones([num_imgs])
        
        #current = self.total_batches[0]
        #self.state = self.state + 1
        #state = self.state # for printing
        
        #for i in range(1):
            #current = tf.Print(current, [current], message = "current ")
            #print_state = tf.Print(self.state, [self.state], message = "state ")
        #j = 0
        #for elt in ones:
        #for j in range(self.batch_size):
        j = tf.constant(0, dtype=tf.int32)
        #condition = lambda j: tf.less(j, num_imgs)
        
        def condition(index):
            return tf.less(index, tf.shape(inputs[0])[0])
        
        def small_body(index):
            unary_list.append(tf.transpose(inputs[0][j,:,:,:], perm=(2,0,1)))
            rgb_list.append(tf.transpose(inputs[1][j,:,:,:], perm=(2,0,1)))
            return index+1
        
        res1 = tf.while_loop(condition, small_body, [j])
        #j+=1
        print("past while loop")
        unaries_tensor = tf.stack(unary_list)
        rgb_tensor = tf.stack(rgb_list)

        #tf.while_loop(condition,body,[i])
        #j = 0
        #for j in range(self.batch_size):
        #for elt in ones:
        j = tf.constant(0, dtype=tf.int32)
        #condition = lambda j: tf.less(j, num_imgs)
        def large_body(index):
            #unaries = tf.transpose(inputs[0][j, :, :, :], perm=(2, 0, 1)) # the fcn_scores
            unaries = unaries_tensor[j]
            #rgb = tf.transpose(inputs[1][j, :, :, :], perm=(2, 0, 1)) # the raw rgb
            rgb = rgb_tensor[j]
            
            c, h, w = self.num_classes, self.image_dims[0], self.image_dims[1]
            all_ones = np.ones((c, h, w), dtype=np.float32)

            # Prepare filter normalization coefficients
            spatial_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=False,
                                                              theta_gamma=self.theta_gamma)
            
            bilateral_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
                                                                theta_alpha=self.theta_alpha,
                                                                theta_beta=self.theta_beta)
            q_values = unaries
            #for k in range(1):
            #    q_values = tf.Print(q_values, [q_values[k]], message="q_values[k] ", summarize=5)
            for i in range(self.num_iterations):
                softmax_out = tf.nn.softmax(q_values, 0)
                
                # Spatial filtering
                spatial_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=False,
                                                            theta_gamma=self.theta_gamma)
                spatial_out = spatial_out / spatial_norm_vals
                
                # Bilateral filtering
                bilateral_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=True,
                                                              theta_alpha=self.theta_alpha,
                                                              theta_beta=self.theta_beta)
                bilateral_out = bilateral_out / bilateral_norm_vals

                # Weighting filter outputs
                message_passing = (tf.matmul(self.spatial_ker_weights,
                                             tf.reshape(spatial_out, (c, -1))) +
                                   tf.matmul(self.bilateral_ker_weights,
                                             tf.reshape(bilateral_out, (c, -1))))
                
                # Compatibility transform
                pairwise = tf.matmul(self.compatibility_matrix, message_passing)
                
                # Adding unary potentials
                pairwise = tf.reshape(pairwise, (c, h, w))
                q_values = unaries - pairwise
            q_values_list.append(q_values)
            return index+1
        
        res2 = tf.while_loop(condition, large_body, [j])
        l = tf.stack(q_values_list)
        return tf.transpose(tf.reshape(l, (tf.shape(inputs[0])[0], self.num_classes, self.image_dims[0], self.image_dims[1])), perm=(0, 2, 3, 1))

    def compute_output_shape(self, input_shape):
        return input_shape
