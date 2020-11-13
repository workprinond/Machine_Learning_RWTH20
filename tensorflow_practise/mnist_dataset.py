#!/usr/bin/env python3

import tensorflow as tf


def load_image(id):
  filename = tf.constant("MNIST_images/", tf.string) + tf.as_string(id) + tf.constant(".png")
  img_string = tf.read_file(filename)
  img = tf.image.decode_image(img_string)
  return img


ids = tf.data.Dataset.range(100)
images = ids.map(load_image)
images_rep = images.repeat()
images_batched = images_rep.batch(100)
it = images_rep.make_one_shot_iterator()
next_img = it.get_next()

s = tf.Session()
for n in range(1000):
  img_val = s.run(next_img)
  print(img_val.shape)
