import tensorflow as tf
import numpy as np


with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, [2, 3],name='input')
    x_pad = tf.pad(x,[[0, 1], [1, 1]],"CONSTANT")
    
    sess=tf.InteractiveSession()  
    sess.run(tf.global_variables_initializer())
    
    data = np.array([[1, 2, 3], [4, 5, 6]])
    y_val = sess.run(x_pad, feed_dict={x: data})
    
    print y_val