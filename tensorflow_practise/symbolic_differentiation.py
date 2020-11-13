#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

s = tf.Session()
x = tf.placeholder(tf.float32, ())
y = x ** 3
#automatically compute derivative of y wrt. x
z = tf.gradients([y], [x])

feed_dict = {x: 5.0}
z_val = s.run(z, feed_dict)
print(z_val)
#expected result:
#y = x ** 3 => z = dy/dx = 3 * (x ** 2) = 3 * (5 ** 2) = 3 * 25 = 75
assert np.isclose(z_val, 75)
