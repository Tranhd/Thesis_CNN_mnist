# model.py
# The CNN-classifier to be supervised on the Mnist dataset


# Imports
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


class MnistCNN(object):
    def __init__(self, sess, save_dir='./MnistCNN_save/'):
        """
        Init-function of the Mnist CNN class

        :param sess: Tensorflow session
        :param save_dir: String
            Save directory for the graph
        """
        self.sess = sess  # Assign Tensorflow session to model.
        self.save_dir = save_dir
        self.build_model()  # Build the graph.

    def network(self, input):
        """
        Predicts the class of test_image

        :param input: numpy array
            The test_images to be classified [n_examples, 28, 28 ,1]

        :return predictions: numpy array
            The predicted class for every example in test_image [n_examples, 1]:
        :return probs: numpy array
            Probability distribution for the predictions [n_examples, 10]

        :raises: Exception
            When model has no checkpoint and weights to load.
        """
        with tf.variable_scope('Network'):

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(
                inputs=input,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            self.activations.append(conv1)
            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            self.activations.append(conv2)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            # Dense Layer
            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            self.activations.append(dense)
            dropout = tf.layers.dropout(
                inputs=dense, rate=0.4, training=self.training)

            # Logits Layer
            logits = tf.layers.dense(inputs=dropout, units=10)
            self.activations.append(logits)
            predictions = tf.nn.softmax(logits)
            return predictions, logits

    def build_model(self):
        """
        Builds the Tensorflow graph

        """
        # Placeholders
        with tf.variable_scope('Placeholders'):
            self.inputs = tf.placeholder(tf.float32, [None, 28, 28, 1])  # Mnist input.
            self.labels = tf.placeholder(tf.int16, [None, 10])  # Labels, one-hot encoded.
            self.training = tf.placeholder(tf.bool)  # Bool indicating if in training mode.
            self.learning_rate = tf.placeholder(tf.float32)  # Learning rate.

        # Activations from CNN
        with tf.variable_scope('Activations'):
            self.activations = list()

        # Predictions and Logits
        with tf.variable_scope('Predictions'):
            self.predictions, self.logits = self.network(self.inputs)  # Builds network

        # Loss.
        with tf.variable_scope('Loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)  # Cross entropy loss
            self.loss = tf.reduce_mean(cross_entropy)  # Loss

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)  # Optimizer

        self.saver = tf.train.Saver()  # Saver

    def train_model(self, x_train, y_train, x_val, y_val, batch_size=64, epochs=100, learning_rate=1e-2, verbose=1):
        """
        Trains the model

        :param x_train: numpy array
            Training set input data [batch_size, 28, 28, 1]
        :param y_train: y_train : numpy array
            Training set labels [batch_size, 10] (one-hot encoded)
        :param x_val: x_val : numpy array
            Validation set input data [batch_size, 28, 28, 1]
        :param y_val: y_val : numpy array
            Validation set labels [batch_size, 10] (one-hot encoded)
        :param batch_size: batch_size (optional): int
            Batch size
        :param epochs: epochs (optional): int
            Epochs to run
        :param learning_rate: learning_rate (optional): float
            Learning rate
        :param verbose: verbose (optional): binary 0 or 1
            Specifies level of Info
        """
        N = len(x_train) // batch_size # Number of iterations per epoch
        try:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir))  # Restore if checkpoint exists.
        except:
            self.sess.run(tf.global_variables_initializer())  # Otherwise initialize.
        print('Starting training ...')
        for epoch in range(epochs):
            idx = np.random.permutation(len(x_train))
            x, y = x_train[idx], y_train[idx]  # Shuffle training data.
            if verbose: print('='*30 + f' Epoch {epoch+1} ' + '='*30)
            loss = 0
            batch_start = 0
            batch_end = batch_size
            for i in range(N):
                if batch_end <= len(x):
                    x_batch, y_batch = x[batch_start:batch_end, :, :, :], y[batch_start:batch_end, :]  # Create batch.
                    _, loss_ = self.sess.run([self.optimizer, self.loss],
                                             feed_dict={self.inputs: x_batch, self.labels: y_batch, self.learning_rate: learning_rate,
                                                          self.training: True})  # Optimize parameters for batch.
                    loss = loss + loss_  # Add to total epoch loss.
                    batch_start = batch_end  # Next batch.
                    batch_end = batch_end + batch_size
            if verbose: print(f'Average Training loss {loss/N}')  # Print average training loss for epoch.

            validation_loss = self.loss.eval(session=self.sess,
                                             feed_dict={self.inputs: x_val, self.labels: y_val, self.training: False})  # Evaluate on validation set.
            if verbose: print(f'Validation loss {validation_loss}')  # Print validation loss for epoch
        self.saver.save(self.sess, save_path=self.save_dir + 'Cnn_mnist.ckpt')  # Save parameters.

    def predict(self, test_image):
        """
        Predicts the class of test_image

        :param test_image: numpy array
            The test_images to be classified [n_examples, 28, 28 ,1]

        :return predictions: numpy array
            The predicted class for every example in test_image [n_examples, 1]
        :return probs: numpy array
            Probability distribution for the predictions [n_examples, 10]
        :return activations : list
            List with activations from each layer of the network, entry i contains
            activations from layer i, numpy array of shape [n_examples, feature map size of layer i]

        :raises Exception
            When model has no checkpoint and weights to load.
        """
        try:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir))  # Try to restore weights.
        except:
            raise Exception(f'Train the model before testing, cant find checkpoint in {self.save_dir}')  # Otherwise => Exception.

        probs, activations = self.sess.run([self.predictions, self.activations],
                                           feed_dict={self.inputs: test_image, self.training: False})  # Predict.

        predictions = np.argmax(probs, axis=1)
        return predictions, probs, activations


# Load MNIST and, if omniglot_bool, Omniglot datasets.
mnist = input_data.read_data_sets('MNIST_data', validation_size=5000, reshape=False, one_hot= True)

# Build model.
tf.reset_default_graph()
sess = tf.Session()
net = MnistCNN(sess)

# Train model.
#net.train_model(mnist.train.images, mnist.train.labels, mnist.validation.images, mnist.validation.labels, epochs=1, verbose=1)

# Test model.

x_test = mnist.test.images[1:1000,:,:,:]
y_test = mnist.test.labels[1:1000,:]
preds, _, activations = net.predict(x_test)
print(np.shape(y_test))
print(np.shape(x_test))

# Evaluate with accuracy.
accuracy = np.sum(np.argmax(y_test, 1) == preds)
print(f'Test accuracy {accuracy*100/len(y_test)} %')

#for i in range(len(activations)):
#    print(np.shape(activations[i]))

class_activations = list()
for cl in range(10):
    i = (np.argmax(y_test,1) == preds)
    i = i & (preds == cl)
    act = activations[-1][i,:]
    class_activations.append(act)
    print(np.shape(act))


rows, cols = 10,1
fix, axes = plt.subplots(figsize=(12,5), nrows=rows, ncols=cols, squeeze=False)
k = 0
for ax_row in axes:
    for ax in ax_row:
        ax.imshow((class_activations[k]), aspect='auto')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        k = k+1

plt.show()