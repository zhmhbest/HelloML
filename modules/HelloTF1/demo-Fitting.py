# import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt
from zhmh.tf import generate_network, get_l2_build_lambda

"""
    生成关联数据
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
DATA_SIZE = 600
INPUT_SIZE = 2
OUTPUT_SIZE = 1
x_train_buffer = []
y_train_buffer = []
for i in range(DATA_SIZE):
    _x1_ = np.random.uniform(-1, 1)
    _x2_ = np.random.uniform(0, 2)
    _y_ = 0 if _x1_ ** 2 + _x2_ ** 2 <= 1 else 1
    x_train_buffer.append([np.random.normal(_x1_, 0.1), np.random.normal(_x2_, 0.1)])
    y_train_buffer.append([_y_])

x_train = np.array(x_train_buffer)
y_train = np.array(y_train_buffer)

"""
    定义神经网络
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
LAYER_NEURONS = [INPUT_SIZE, 5, 4, 3, OUTPUT_SIZE]
REGULARIZATION_RATE = 0.003
REGULARIZER_COLLECTION = 'losses'

place_x = tf.placeholder(tf.float32, shape=(None, INPUT_SIZE))
place_y = tf.placeholder(tf.float32, shape=(None, OUTPUT_SIZE))

y = generate_network(
    LAYER_NEURONS, place_x, 1, 0.001,
    get_l2_build_lambda(REGULARIZATION_RATE, REGULARIZER_COLLECTION, tf.nn.elu)
)

"""
    定义损失函数
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
loss_mse = tf.losses.mean_squared_error(place_y, y)
loss_l2 = tf.add_n(tf.get_collection(REGULARIZER_COLLECTION)) + loss_mse


"""
    训练
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
LEARNING_RATE = 0.001
TRAIN_TIMES_UF = 400
TRAIN_TIMES_FI = 8000
train_op_mse = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_mse)
train_op_l2 = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_l2)


def plot_graph(_sess):
    def paint_scatter_data(feature, label, split_line=None, c1='HotPink', c2='DarkCyan'):
        plt.scatter(feature[:, 0], feature[:, 1],
                    c=[c1 if _i_ == 0 else c2 for _i_ in label],
                    cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white")
        if split_line is not None:
            plt.contour(
                split_line['x'], split_line['y'], split_line['probs'],
                levels=[.5], cmap="Greys", vmin=0, vmax=.1
            )
        plt.show()
    # end def
    # 计算分割曲线
    xx, yy = np.mgrid[-1.2:1.2:.01, -0.2:2.2:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = _sess.run(y, feed_dict={place_x: grid}).reshape(xx.shape)
    # print(probs)
    # 画出
    paint_scatter_data(x_train, y_train, {'x': xx, 'y': yy, 'probs': probs})


# 欠拟合

with tf.Session() as sess:
    # 初始化全部变量OP
    tf.global_variables_initializer().run()
    for i in range(1, 1 + TRAIN_TIMES_UF):
        sess.run(train_op_mse, feed_dict={
            place_x: x_train,
            place_y: y_train
        })
    plot_graph(sess)


# 过拟合
with tf.Session() as sess:
    # 初始化全部变量OP
    tf.global_variables_initializer().run()
    for i in range(1, 1 + TRAIN_TIMES_FI):
        sess.run(train_op_mse, feed_dict={
            place_x: x_train,
            place_y: y_train
        })
    plot_graph(sess)

# 加入L2正则化
with tf.Session() as sess:
    # 初始化全部变量OP
    tf.global_variables_initializer().run()
    for i in range(1, 1 + TRAIN_TIMES_FI):
        sess.run(train_op_l2, feed_dict={
            place_x: x_train,
            place_y: y_train
        })
    plot_graph(sess)
