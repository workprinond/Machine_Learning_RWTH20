#!/usr/bin/env python3

import tensorflow as tf

s = tf.Session()
x = tf.placeholder(tf.float32)
y = x ** 2
feed_dict = {x: 5}
y_val = s.run(y, feed_dict)
print(y_val)
