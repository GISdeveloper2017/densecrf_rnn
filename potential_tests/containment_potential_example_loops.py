# Containment tem using for loops
import tensorflow as tf
import pdb

s = tf.InteractiveSession()
nb_classes = 3
rows,cols = 4,5
correct_labeling = tf.constant([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [1, 1, 2, 2, 2],
                                [1, 1, 2, 2, 2]])
q_vals = tf.constant([[[0.9, 0.9, 0.9, 0.9, 0.9],
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
bd_map = tf.constant([[1,1,1,2,2],
                      [1,1,1,2,2],
                      [3,3,4,4,5],
                      [3,3,4,4,5]])


# replicate  m times and have the shape of [rows,cols,l] where l is the number of labels
extended_bd_map = tf.stack([bd_map] * nb_classes)
# Get number of cliques
flat = tf.reshape(bd_map, [-1])
y, index = tf.unique(flat)
num_cliques = s.run(tf.size(y))
values = [tf.cast(tf.equal(bd_map,i), tf.float32) for i in range(1,length+1)]
split_bd_map = tf.stack(values)
bool_bd_map = tf.stack([tf.equal(bd_map,i) for i in range(1,length+1)])

# This will put True where the max prob label, False otherwise:
bool_max_label = tf.equal(q_vals, tf.reduce_max(q_vals,axis=0))

# These would be the learned parameters:
w_low = tf.constant(0.1)
w_low_m = tf.constant([[0.11,0.,0.],
                       [0., 0.10, 0.],
                       [0., 0., 0.09]])
w_low_m_1d = tf.constant([0.11,0.10,0.9])
w_low_m_1d_duplicated = tf.stack([w_low_m_1d]*(rows*cols))
w_high = tf.constant(0.9)

for l in range(nb_classes):
    for i in range(rows):
        for j in range(cols):
            # clique index is value of (i,j) in bd_map
            clique_index = bd_map[i][j]
            product_matrix = tf.multiply(split_bd_map[clique_index-1], q_vals[l])
            val = s.run(product_matrix[i][j])
            l_matrix = tf.multiply(split_sp_map[clique_index-1], q_vals[l_prime])
            flattened = tf.reshape(product_matrix, [-1])
            flattened_l = tf.reshape(product_matrix, [-1])
            flattened = tf.add(flattened, flattened_l)
            # Want the product of all nonzero elements in flattened
            zero = tf.constant(0, dtype=tf.float32)
            where_nonzero = tf.not_equal(flattened, zero)
            reduced = tf.boolean_mask(flattened, where_nonzero)
            product = tf.reduce_prod(reduced)
            # Must divide product by value at q_i = l
            product = tf.divide(product, val)
            op = bd_out[l,i,j].assign(bd_out[l,i,j]+product)
            s.run(op)
print(s.run(bd_out))

# the actual product: we need to divide it by each index's q_val(r,c,l) + q_val(r,c,l')
first_term = tf.divide(tf.to_float(prod_tensor),q_val_sum_tensor)

# multiply by weights
#first_term_resp = tf.matmul(w_low_m,tf.reshape(first_term, (nb_classes,-1)))
first_term_resp = tf.multiply(tf.transpose(w_low_m_1d_duplicated),tf.reshape(first_term, (nb_classes,-1)))

first_term_resp_back = tf.reshape(first_term_resp, (nb_classes, rows, cols))

containment_out = first_term_resp_back + w_high * (tf.ones(shape=first_term_resp_back.shape) - first_term_resp_back)

print(s.run(containment_out))

    
