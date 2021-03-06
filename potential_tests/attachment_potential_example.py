from __future__ import division
import tensorflow as tf
import numpy as np

def get_avg_clique_size(sp_map):
    """ Computes the average size in pixels of a clique
    """
    unique = max(np.unique(sp_map))
    clique_sizes = [np.size(np.where(unique == i)) for i in range(1,unique+1)]
    return 10*np.mean(clique_sizes), unique

def get_center(tup):
    """ Get center of ([y_indices], [x_indices])
    """
    length = len(tup[0])
    sum_x = sum(tup[0])
    sum_y = sum(tup[1])
    return sum_x/length, sum_y/length

def get_close_sp(sp_map):
    """ Return dictionary with c1: [c2, ... , cn] mapping clique numbers to their neighboring cliques
    """
    threshold, unique = get_avg_clique_size(sp_map)
    indices = [np.nonzero(sp_map == i) for i in range(1,unique+1)]
    centers = [get_center(tup) for tup in indices]
    center_dict = dict()
    for i in range(1,unique+1):
        center_dict[i] = []
        for j in range(i+1,unique+1):
            if np.linalg.norm(np.array(centers[j-1]) - np.array(centers[i-1])) < threshold:
                center_dict[i].append(j)
    return center_dict
    
s = tf.InteractiveSession()
c,h,w = 3,4,5
correct_labeling = tf.constant([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [1, 1, 2, 2, 2],
                                [1, 1, 2, 2, 2]])
q_values = tf.constant([[[0.9, 0.9, 0.9, 0.9, 0.9],
                       [0.9, 0.9, 0.9, 0.9, 0.9],
                       [0.01, 0.01, 0.01, 0.01, 0.01],
                       [0.01, 0.01, 0.01, 0.01, 0.01]],
                      [[0.01, 0.01, 0.01, 0.01, 0.01],
                       [0.01, 0.01, 0.01, 0.01, 0.01],
                       [0.9, 0.9, 0.8, 0.7, 0.6],
                       [0.9, 0.9, 0.81, 0.5, 0.4]],
                      [[0.01, 0.01, 0.01, 0.01, 0.01],
                       [0.01, 0.01, 0.01, 0.01, 0.01],
                       [0.4, 0.5, 0.9, 0.9, 0.9],
                       [0.7, 0.8, 0.8, 0.9, 0.9]]])

q_vals_arr = [[[0.9, 0.9, 0.9, 0.9, 0.9],
                       [0.9, 0.9, 0.9, 0.9, 0.9],
                       [0.01, 0.01, 0.01, 0.01, 0.01],
                       [0.01, 0.01, 0.01, 0.01, 0.01]],
                      [[0.01, 0.01, 0.01, 0.01, 0.01],
                       [0.01, 0.01, 0.01, 0.01, 0.01],
                       [0.9, 0.9, 0.8, 0.7, 0.6],
                       [0.9, 0.9, 0.81, 0.5, 0.4]],
                      [[0.01, 0.01, 0.01, 0.01, 0.01],
                       [0.01, 0.01, 0.01, 0.01, 0.01],
                       [0.4, 0.5, 0.9, 0.9, 0.9],
                       [0.7, 0.8, 0.8, 0.9, 0.9]]]
sp_map = tf.constant([[1,1,1,2,2],
                      [1,1,1,2,2],
                      [3,3,4,4,5],
                      [3,3,4,4,5]])
sp_map_arr = np.array([[1,1,1,2,2],
                      [1,1,1,2,2],
                      [3,3,4,4,5],
                      [3,3,4,4,5]])

extended_sp_map = tf.stack([sp_map]*c) #shape (h,w,c)

# Get number of cliques as a scalar
flat = tf.reshape(sp_map, [-1])
y, index = tf.unique(flat)
length = s.run(tf.size(y))

# Initialize variables to change them
attachment_out = tf.get_variable("attachment_out", [c,h,w], dtype=tf.float32, initializer=tf.zeros_initializer)
s.run(tf.global_variables_initializer())

# Weights
w_low = tf.constant([[0.1, 0.1, 0.1], # nb_classes x nb_classes
                     [0.1, 0.1, 0.1],
                     [0.1, 0.1, 0.1]])
w_high = tf.constant(0.9)

prod_tensor = tf.zeros(shape=(c,h,w))
# Use a dictionary to tell which pairs of sp constitute a clique
distance_dict = get_close_sp(sp_map_arr)
for l1 in range(1,length+1):
    # Get locations of first sp in clique
    bool_sp_indx1 = tf.equal(extended_sp_map,l1)
    for l2 in distance_dict[l1]:
        bool_sp_indx2 = tf.equal(extended_sp_map, l2)
        # Put 1 in q_values if doesn't belong to this clique
        A1 = tf.multiply(tf.to_float(bool_sp_indx1), q_values) + tf.to_float(tf.logical_not(bool_sp_indx1))
        A2 = tf.multiply(tf.to_float(bool_sp_indx2), q_values) + tf.to_float(tf.logical_not(bool_sp_indx2))
        # Compute product for each cell:
        B1 = tf.reduce_prod(A1, [1,2])
        B2 = tf.reduce_prod(A2, [1,2])
        # Create tensor containing products for each cell
        C1 = tf.stack([B1]*(h*w))
        C1 = tf.reshape(tf.transpose(C1), (c,h,w))
        C1 = tf.multiply(tf.to_float(bool_sp_indx1), C1)
        C2 = tf.stack([B2]*(h*w))
        C2 = tf.reshape(tf.transpose(C2), (c,h,w))
        C2 = tf.multiply(tf.to_float(bool_sp_indx2), C2)
        # Add to overall product
        prod_tensor += C1 + C2
        
# TODO: remove current q_value and don't add 0.9 if sp wasn't part of a clique
first_term = tf.divide(tf.to_float(prod_tensor),q_values)
first_term_resp = tf.matmul(w_low,tf.reshape(first_term, (c,-1)))
first_term_resp_back = tf.reshape(first_term_resp, (c, h, w))
attachment_out =  first_term_resp_back + w_high * (tf.ones(shape=(c,h,w)) - first_term)
print(s.run(attachment_out))
