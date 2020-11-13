#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

s = tf.Session()
W = tf.get_variable("W", shape=(), dtype=tf.float32, initializer=tf.constant_initializer(5.0))
x = tf.constant(3.0, tf.float32)
y = W * x

#variables need to be initialized!
s.run(tf.global_variables_initializer())

y_val = s.run(y)
#y should now be 5, so W * x = 5 * 3 = 15, let's check
print(y_val)
assert np.isclose(y_val, 15)

#let's change the value of W
W.assign(20.0)
#let's check if it worked
W_val = s.run(W)
print(W_val)

#what happened?
#remember: assign is just a graph node created by tensorflow, we still need to execute it!
assign_op = W.assign(20.0)
s.run(assign_op)
#let's check if it worked this time
W_val = s.run(W)
print(W_val)
assert W_val == 20.0
