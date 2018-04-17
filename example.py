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
train_loss, train_acc, val_loss, val_acc = net.train_model(mnist.train.images, mnist.train.labels,
                mnist.validation.images, mnist.validation.labels, epochs=3, verbose=1)

# Test model.
preds, _, activations = net.predict(mnist.test.images)

# Evaluate with accuracy.
accuracy = np.sum(np.argmax(mnist.test.labels, 1) == preds)
print(f'Test accuracy {accuracy/len(mnist.test.labels)} %')

for i in range(len(activations)):
    print(f'Activations-shape of layer {i+1}: {np.shape(activations[i])}')