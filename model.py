# model.py
# The CNN-classifier to be supervised using the Mnist dataset


# Imports
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class MnistCNN(object):
    def __init__(self, sess, save_dir='./MnistCNN_save/', log_dir='./logs/'):
        """
        Init-function of the Mnist CNN class

        Parameters
        ----------
        sess : Tensorflow session
        save_dir (optional): String
            Save directory for the graph
        log_dir (optional): String
            Where to save tensorboard log files

        """
        self.sess = sess # Assign Tensorflow session to model.
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.build_model() # Build the graph.


    def build_model(self):
        """
        Builds the tensorflow graph
        
        """
        # Placeholders
        with tf.variable_scope('Placeholders'):
            self.inputs = tf.placeholder(tf.float32, [None, 28, 28, 1]) # Mnist input.
            self.labels = tf.placeholder(tf.int16, [None, 10]) # Labels, one-hot encoded.
            self.training = tf.placeholder(tf.bool) # Bool indicating if in training mode.
            self.learning_rate = tf.placeholder(tf.float32) # Learning rate.

        # Activations from CNN
        with tf.variable_scope('Activations'):
            self.activations = list()

        # Predictions and Logits
        with tf.variable_scope('Predictions'):
            self.predictions, self.logits = self.network(self.inputs) # Builds network

        # Loss.
        with tf.variable_scope('Loss')
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits) # Cross entropy loss
            self.loss = tf.reduce_mean(cross_entropy) # Loss

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss) # Optimizer

        self.saver = tf.train.Saver() # Saver


    def train_model(self, x_train, y_train, x_val, y_val, batch_size=64, epochs=100, learning_rate=1e-2, verbose=1):
        """
        Train the model.

        Parameters
        ----------
        x_train : numpy array
            Training set input data [batch_size, 28, 28, 1]
        y_train : numpy array
            Training set labels [batch_size, 10] (one-hot encoded)
        x_val : numpy array
            Validation set input data [batch_size, 28, 28, 1]
        y_val : numpy array
            Validation set labels [batch_size, 10] (one-hot encoded)
        batch_size (optional): int
            Batch size
        epochs (optional): int
            Epochs to run
        learning_rate (optional): float
            Learning rate
        verbose (optional): binary 0 or 1
            Specifies level of Info
        """
        N = len(x_train) // batch_size # Number of iterations per epoch
        try:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir)) # Restore if checkpoint exists.
        except:
            self.sess.run(tf.global_variables_initializer()) # Otherwise initialize.
        print('Starting training ...')
        for epoch in range(epochs):
            idx = np.random.permutation(len(x_train))
            x, y = x_train[idx], y_train[idx] # Shuffle training data.
            if verbose: print('='*30 + f' Epoch {epoch+1} ' + '='*30)
            loss = 0
            batch_start = 0
            batch_end = batch_size
            for i in range(N):
                if batch_end <= len(x):
                    x_batch, y_batch = x[batch_start:batch_end, :, :, :], y[batch_start:batch_end, :] # Create batch.
                    _, loss_ = self.sess.run([self.optimizer, self.loss],
                                             feed_dict = {self.inputs: x_batch, self.labels: y_batch, self.learning_rate: learning_rate,
                                                          self.training: True}) # Optimize parameters for batch.
                    loss = loss + loss_ # Add to total epoch loss.
                    batch_start = batch_end  # Next batch.
                    batch_end = batch_end + batch_size
            if verbose: print(f'Average Training loss {loss/N}') # Print average training loss for epoch.

            validation_loss = self.loss.eval(session=self.sess,
                                             feed_dict={self.inputs: x_val, self.labels: y_val, self.training: False}) # Evaluate on validation set.
            if verbose: print(f'Validation loss {validation_loss}') # Print validation loss for epoch
        self.saver.save(sess, save_path=self.save_dir + 'Cnn_mnist.ckpt') # Save parameters.


    def predict(self, test_image):
        try:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir))
        except:
            raise Exception(f'Train the model before testing, cant find checkpoint in {self.save_dir}')

        probs, activations = self.sess.run([self.predictions, self.activations], feed_dict={self.inputs: test_image, self.training: False})

        predictions = np.argmax(probs, axis=1)
        return predictions, probs, activations

    def network(self, input):
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



mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=False, validation_size=5000)
x_train = mnist.train.images
y_train = mnist.train.labels
x_val = mnist.validation.images
y_val = mnist.validation.labels
x_test = mnist.test.images
y_test = mnist.test.labels
print(np.shape(x_test))

tf.reset_default_graph()
sess = tf.Session()
net = MnistCNN(sess)

#net.train_model(x_train, y_train, x_val, y_val, epochs=50, verbose=1)

preds, _, activations = net.predict(x_test)
accuracy = np.sum(np.argmax(y_test, 1) == preds)
print(f'Test accuracy {accuracy/100} %')

print(np.shape(x_test))
for i in range(len(activations)):
    print(np.shape(activations[i]))


































































































































































































































