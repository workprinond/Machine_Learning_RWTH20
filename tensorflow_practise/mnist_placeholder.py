#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#download and extract data from https://omnomnom.vision.rwth-aachen.de/data/mnist.tgz
print("loading data...")
x_data = np.genfromtxt("MNIST_data/mnist-train-data.csv")
y_data = np.genfromtxt("MNIST_data/mnist-train-labels.csv")
print("done.")
print("x_data.shape", x_data.shape)
print("y_data.shape", y_data.shape)

#let's look at the first image
img = x_data[0, :].reshape(28, 28)
plt.imshow(img)
plt.show()

#simple example: let's say we want to compute the average fraction of background pixels (value=0)
#in the first 10000 examples
#assume our memory is limited, so let's work with mini-batches of 100 images
batch_size = 100
x = tf.placeholder(tf.float32, [batch_size, 784])
#count fraction of background pixels
is_background_pixel = tf.equal(x, 0)
#cast from bool to float
is_background_pixel_float = tf.cast(is_background_pixel, tf.float32)
#shape: (batch_size, 784) -> take mean over the 784 pixels
background_fraction = tf.reduce_mean(is_background_pixel_float, axis=1)

s = tf.Session()
background_fractions = []
#divide data into mini batches
n_total = 10000
n_steps = n_total // batch_size
for n in range(n_steps):
  start = n * batch_size
  end = start + batch_size
  x_minibatch = x_data[start:end]
  feed_dict = {x: x_minibatch}
  background_fraction_val = s.run(background_fraction, feed_dict)
  background_fractions.append(background_fraction_val)

background_fractions = np.array(background_fractions)
background_fraction = background_fractions.mean()
print("background fraction", background_fraction * 100, "%")
