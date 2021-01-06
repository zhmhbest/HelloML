import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

a = tf.constant(1)
b = tf.constant(2)
c = tf.add(a, b)

with tf.Session() as sess:
    r = sess.run(c)
    print(r)
