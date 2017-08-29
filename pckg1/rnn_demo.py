'''
Created on Aug 28, 2017

@author: Amin
'''
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
# // integer division
num_batches = total_series_length//batch_size//truncated_backprop_length


def generateData():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)

if __name__ == '__main__':
    print ('in main')
    
    a = tf.placeholder(tf.float32,shape=(1,))
    b = tf.constant(12)
    #c = a*b
    
    sess = tf.Session()
    print (sess.run(a,feed_dict={a: [1]}))
    sess.close()