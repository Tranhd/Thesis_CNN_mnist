import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


class MnistCNN(object):
    def __init__(self, sess, save_dir='./MnistCNN/', log_dir='./logs/'):
        self.sess = sess
        self.build_model()
        self.save_dir = save_dir
        self.log_dir = log_dir

    def build_model(self):
        with tf.variable_scope('Placeholders'):
            self.inputs = tf.placeholder(tf.float32, [None, 28, 28, 1])
            self.labels = tf.placeholder(tf.int16, [None, 10])
            self.training = tf.placeholder(tf.bool)
            self.learning_rate = tf.placeholder(tf.float32)
            self.predictions, self.logits = self.network(self.inputs)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            self.saver = tf.train.Saver()


    def train_model(self, x_train, y_train, x_val, y_val, batch_size=64, epochs=100, learning_rate=1e-2, verbose=1):
        N = len(x_train) // batch_size
        try:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir))
        except:
            self.sess.run(tf.global_variables_initializer())
        print('Starting training ...')
        for epoch in range(epochs):
            idx = np.random.permutation(len(x_train))
            x, y = x_train[idx], y_train[idx]
            if verbose: print('='*30 + f' Epoch {epoch+1} ' + '='*30)
            loss = 0
            batch_start = 0
            batch_end = batch_size
            for i in range(N):
                if batch_end <= len(x):
                    x_batch, y_batch = x[batch_start:batch_end, :, :, :], y[batch_start:batch_end, :]
                    _, loss_ = self.sess.run([self.optimizer, self.loss],
                                             feed_dict = {self.inputs: x_batch, self.labels: y_batch, self.learning_rate: learning_rate,
                                                          self.training: True})
                    loss = loss + loss_
                    batch_start = batch_end
                    batch_end = batch_end + batch_size
            if verbose: print(f'Average Training loss {loss/N}')
            validation_loss = self.loss.eval(session=self.sess, feed_dict={self.inputs: x_val, self.labels: y_val, self.training: False})
            if verbose: print(f'Validation loss {validation_loss}')
            self.saver.save(sess, save_path=self.save_dir + 'Cnn_mnist.ckpt')

    def predict(self, test_image):
        try:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir))
        except:
            raise Exception(f'Train the model before testing, cant find checkpoint in {self.save_dir}')

        preds = self.predictions.eval(session = self.sess, feed_dict={self.inputs: test_image, self.training: False})
        return preds

    def network(self, input):
        with tf.variable_scope('Network'):

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(
                inputs=input,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)

            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            # Dense Layer
            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            dropout = tf.layers.dropout(
                inputs=dense, rate=0.4, training=self.training)

            # Logits Layer
            logits = tf.layers.dense(inputs=dropout, units=10)
            predictions = tf.nn.softmax(logits)

            return predictions, logits



mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=False, validation_size=5000)
x_train = mnist.train.images
y_train = mnist.train.labels
x_val = mnist.validation.images
y_val = mnist.validation.labels
#print(np.shape(x_train[1:100,:,:,:]))

tf.reset_default_graph()
sess = tf.Session()
net = MnistCNN(sess)
net.train_model(x_train[0:1000,:,:,:], y_train[0:1000,:], x_val, y_val, epochs=2, verbose=1)
net.predict(x_train)
