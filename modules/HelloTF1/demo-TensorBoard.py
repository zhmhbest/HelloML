# import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from zhmh.tf import TensorBoard

a = tf.constant(0.01, name='a')
b = tf.constant(0.02, name='b')
c = tf.add(a, b, name='c')

tb = TensorBoard()
with tf.Session() as sess:
    print('c =', sess.run(c))
    tb.save(sess.graph)

tb.board()
