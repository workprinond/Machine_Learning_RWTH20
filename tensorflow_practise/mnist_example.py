#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import time
import pickle


def load_data():
  print("loading data...")
  #for the first run uncomment these lines, afterwards just use the pickle file
  # download and extract data from https://omnomnom.vision.rwth-aachen.de/data/mnist.tgz
  #x_train = np.genfromtxt("MNIST_data/mnist-train-data.csv")
  #y_train = np.genfromtxt("MNIST_data/mnist-train-labels.csv")
  #x_valid = np.genfromtxt("MNIST_data/mnist-valid-data.csv")
  #y_valid = np.genfromtxt("MNIST_data/mnist-valid-labels.csv")
  #write it to a pickle file from which we can read much faster than from csv
  #pickle.dump((x_train, y_train, x_valid, y_valid), open("MNIST_data/mnist_data.pkl", "wb"))

  x_train, y_train, x_valid, y_valid = pickle.load(open("MNIST_data/mnist_data.pkl", "rb"))
  print("done.")

  #normalize data to [0,1]
  x_train /= 255
  x_valid /= 255

  return x_train, y_train, x_valid, y_valid


def create_logistic_regression_model(x):
  W = tf.get_variable("W", (28*28, 10), tf.float32, initializer=tf.zeros_initializer)
  b = tf.get_variable("b", (10,), tf.float32, initializer=tf.zeros_initializer)
  logits = tf.matmul(x, W) + b
  return logits


def create_fully_connected_model(x):
  #let's add a hidden layers with 100 hidden units and ReLU
  #use (Xavier) Glorot initialization
  n_hidden = 100
  W_hidden = tf.get_variable("W_hidden", (28 * 28, n_hidden), tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
  b_hidden = tf.get_variable("b_hidden", (n_hidden,), tf.float32,
                             initializer=tf.zeros_initializer)
  z = tf.matmul(x, W_hidden) + b_hidden
  h = tf.nn.relu(z)

  #add another hidden layer
  #n_hidden2 = 100
  #W_hidden2 = tf.get_variable("W_hidden2", (n_hidden, n_hidden2), tf.float32,
  #                            initializer=tf.contrib.layers.xavier_initializer())
  #b_hidden2 = tf.get_variable("b_hidden2", (n_hidden2,), tf.float32,
  #                            initializer=tf.zeros_initializer)
  #z = tf.matmul(h, W_hidden2) + b_hidden2
  #h = tf.nn.relu(z)

  #add output layer as before
  n_out = 10
  W_out = tf.get_variable("W_out", (n_hidden, n_out), tf.float32, initializer=tf.zeros_initializer)
  b_out = tf.get_variable("b_out", (n_out,), tf.float32, initializer=tf.zeros_initializer)
  logits = tf.matmul(h, W_out) + b_out
  return logits


def create_conv_layer(x, n_input, n_output):
  # create a 5x5 filter kernel
  W_conv = tf.get_variable("W_hidden", (5, 5, n_input, n_output), tf.float32,
                           initializer=tf.contrib.layers.xavier_initializer_conv2d())
  # apply convolution and ReLU
  z = tf.nn.conv2d(x, W_conv, strides=[1, 1, 1, 1], padding="SAME")
  h = tf.nn.relu(z)
  return h


def create_convnet_model(x):
  #reshape data back to an image
  x = tf.reshape(x, [-1, 28, 28, 1])

  with tf.variable_scope("conv1"):
    h = create_conv_layer(x, n_input=1, n_output=16)
  h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
  with tf.variable_scope("conv2"):
    h = create_conv_layer(h, n_input=16, n_output=32)
  h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
  with tf.variable_scope("conv3"):
    h = create_conv_layer(h, n_input=32, n_output=64)
  h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
  with tf.variable_scope("conv4"):
    h = create_conv_layer(h, n_input=64, n_output=128)
  h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

  #collapse to vector
  batch_size = tf.shape(h)[0]
  height = h.get_shape()[1]
  width = h.get_shape()[2]
  n_features = h.get_shape()[3]
  n_features_collapsed = height * width * n_features
  collapsed_shape = tf.stack([batch_size, n_features_collapsed])
  h = tf.reshape(h, collapsed_shape)

  # add output layer as before
  n_out = 10
  W_out = tf.get_variable("W_out", (n_features_collapsed, n_out), tf.float32, initializer=tf.zeros_initializer)
  b_out = tf.get_variable("b_out", (n_out,), tf.float32, initializer=tf.zeros_initializer)
  logits = tf.matmul(h, W_out) + b_out
  return logits


def create_loss(logits, y_ref):
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_ref)
  loss_mean = tf.reduce_mean(loss, axis=0)
  return loss_mean


def create_errors(logits, y_ref):
  pred = tf.argmax(logits, axis=1)
  errs = tf.not_equal(pred, y_ref)
  errs_float = tf.cast(errs, tf.float32)
  errs_mean = tf.reduce_mean(errs_float)
  return errs_mean


def create_train_op(loss, global_step):
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def run_epoch(sess, train_step, loss, errs, x, y, x_data, y_data, summary_writer, summaries, global_step):
  batch_size = 100
  n_total = x_data.shape[0]
  n_steps = n_total // batch_size

  total_loss = 0.0
  total_errs = 0
  #shuffle data randomly
  perm = np.random.permutation(x_data.shape[0])
  x_data_shuff = x_data[perm]
  y_data_shuff = y_data[perm]
  for n in range(n_steps):
    start = n * batch_size
    end = (n + 1) * batch_size
    x_batch = x_data_shuff[start:end]
    y_batch = y_data_shuff[start:end]
    feed_dict = {x: x_batch, y: y_batch}
    loss_val, errs_val, _, summ_val, global_step_val = sess.run([loss, errs, train_step, summaries, global_step],
                                                                feed_dict)
    total_loss += loss_val
    total_errs += errs_val
    if summary_writer is not None:
      summary_writer.add_summary(summ_val, global_step_val)
  avg_loss = total_loss / n_steps
  avg_errs = total_errs / n_steps
  return avg_loss, avg_errs


def create_summaries(loss, errs, x, name=""):
  loss_summ = tf.summary.scalar("loss", loss)
  errs_summ = tf.summary.scalar("errs", errs)
  img_summ = tf.summary.image("imgs", tf.reshape(x, [-1, 28, 28, 1]))
  summ_merged = tf.summary.merge([loss_summ, errs_summ, img_summ])
  summary_writer = tf.summary.FileWriter("summaries/" + name)
  #for simplicity let's only use summaries on the training set
  summaries_valid = tf.no_op()
  return summary_writer, summ_merged, summaries_valid


def save_model(sess, path, global_step):
  saver = tf.train.Saver()
  saver.save(sess, path, global_step)


def load_model(sess, path):
  saver = tf.train.Saver()
  saver.restore(sess, path)


def main():
  sess = tf.Session()
  x_train, y_train, x_valid, y_valid = load_data()
  x = tf.placeholder(tf.float32, [None, 28*28])
  y = tf.placeholder(tf.int64, [None])
  #logits = create_logistic_regression_model(x)
  #logits = create_fully_connected_model(x)
  logits = create_convnet_model(x)
  loss = create_loss(logits, y)
  errs = create_errors(logits, y)
  #variable which counts the number of update steps done so far
  global_step = tf.train.create_global_step()
  train_op = create_train_op(loss, global_step)
  validation_step = tf.no_op()
  sess.run(tf.global_variables_initializer())
  summary_writer, summaries_train, summaries_valid = create_summaries(loss, errs, x)
  #write the graph out
  summary_writer.add_graph(tf.get_default_graph())
  n_epochs = 20
  #load_model(sess, "models/logistic_regression-5000")
  for epoch in range(n_epochs):
    start = time.time()
    train_loss, train_errs = run_epoch(sess, train_op, loss, errs, x, y, x_train, y_train, summary_writer,
                                       summaries_train, global_step)
    valid_loss, valid_errs = run_epoch(sess, validation_step, loss, errs, x, y, x_valid, y_valid, None,
                                       summaries_valid, global_step)
    end = time.time()
    elapsed = end - start
    print("epoch:", epoch + 1, "elapsed", elapsed, "train loss:", train_loss, "train errors:", train_errs,
          "valid loss:", valid_loss, "valid errors:", valid_errs)
  #save_model(sess, "models/logistic_regression", global_step)


if __name__ == "__main__":
  main()
  #to try:
  #1) look at summaries in tensorboard
  #2) different models (logistic regression, feed forward network, convnet)
  #3) different optimizers, learning rates
