from cnn import MnistCNN
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Load model
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False)

# Build model.
tf.reset_default_graph()
sess = tf.Session()
net = MnistCNN(sess)

# Train model.
net.train_model(mnist.train.images, mnist.train.labels,
                mnist.validation.images, mnist.validation.labels, epochs=1, verbose=1)

# Test model.
preds, _, activations = net.predict(mnist.train.images)

# Evaluate with accuracy.
accuracy = np.sum(np.argmax(mnist.test.labels, 1) == preds)
print(f'Test accuracy {accuracy/len(y_test)} %')

for i in range(len(activations)):
    print(f'Activations-shape of layer {i+1}: {np.shape(activations[i])}')