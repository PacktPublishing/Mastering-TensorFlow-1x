import os

import numpy as np

import tensorflow as tf
tf.set_random_seed(123)
print("TensorFlow:{}".format(tf.__version__))

DATASETSLIB_HOME = '../datasetslib'
import sys
if not DATASETSLIB_HOME in sys.path:
    sys.path.append(DATASETSLIB_HOME)
import datasetslib

datasetslib.datasets_root = os.path.join(os.path.expanduser('~'),'datasets')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(os.path.join(datasetslib.datasets_root,'mnist'), one_hot=True)

x_train = mnist.train.images
x_test = mnist.test.images
y_train = mnist.train.labels
y_test = mnist.test.labels

# parameters
n_y = 10  # 0-9 digits
n_x = 784  # total pixels

def mlp(x, num_inputs, num_outputs,num_layers,num_neurons):
    w=[]
    b=[]
    for i in range(num_layers):
        # weights
        w.append(tf.Variable(tf.random_normal(
                              [num_inputs if i==0 else num_neurons[i-1],
                               num_neurons[i]]),
                             name="w_{0:04d}".format(i)
                            )
                )
        # biases
        b.append(tf.Variable(tf.random_normal(
                              [num_neurons[i]]),
                             name="b_{0:04d}".format(i)
                            )
                )
    w.append(tf.Variable(tf.random_normal(
                          [num_neurons[num_layers-1] if num_layers > 0 else num_inputs,
                           num_outputs]),name="w_out"))
    b.append(tf.Variable(tf.random_normal([num_outputs]),name="b_out"))

    assert_op = tf.Assert(tf.reduce_all(tf.greater_equal(x,0)),[x])
    with tf.control_dependencies([assert_op]):
        # x is input layer
        layer = x
        # add hidden layers
        for i in range(num_layers):
            layer = tf.nn.relu(tf.matmul(layer, w[i]) + b[i])
        # add output layer
        layer = tf.matmul(layer, w[num_layers]) + b[num_layers]

    return layer

num_layers = 2
num_neurons = [16,32]
learning_rate = 0.01
n_epochs = 10
batch_size = 100
n_batches = int(mnist.train.num_examples/batch_size)

# input images
x_p = tf.placeholder(dtype=tf.float32, name="x_p", shape=[None, n_x])
# target output
y_p = tf.placeholder(dtype=tf.float32, name="y_p", shape=[None, n_y])

model = mlp(x=x_p,
            num_inputs=n_x,
            num_outputs=n_y,
            num_layers=num_layers,
            num_neurons=num_neurons)

model = tf.Print(input_=model,
                 data=[tf.argmax(model,1)],
                 message='y_hat=',
                 summarize=10,
                 first_n=5
                )

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y_p))
# optimizer function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = optimizer.minimize(loss)

#predictions_check = tf.equal(tf.argmax(model,1), tf.argmax(y,1))
#accuracy_function = tf.reduce_mean(tf.cast(predictions_check, tf.float32))
from tensorflow.python import debug as tfd

with tfd.LocalCLIDebugWrapperSession(tf.Session()) as tfs:
        tfs.run(tf.global_variables_initializer())
        tfs.add_tensor_filter('has_inf_or_nan_filter', tfd.has_inf_or_nan)
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for batch in range(n_batches):
                X_batch, Y_batch = mnist.train.next_batch(batch_size)
                if epoch > 0:
                    X_batch = np.copy(X_batch)
                    X_batch[0,0]=np.inf
                feed_dict={x_p: X_batch, y_p: Y_batch}
                _,batch_loss = tfs.run([optimizer,loss],
                                       feed_dict = feed_dict
                                      )
                epoch_loss += batch_loss
            average_loss = epoch_loss / n_batches
            print("epoch: {0:04d}   loss = {1:0.6f}".format(epoch,average_loss))
