# import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


def train_constant_rate(demo_name, learning_rate, training_times):
    """
    使用固定学习率训练
    找到一个x使得x^2最小
    :param demo_name: 训练名称
    :param learning_rate: 训练速率
    :param training_times: 训练次数
    :return:
    """
    x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
    # loss = x ^ 2
    loss = tf.square(x)
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(1, 1 + training_times):
            sess.run(train_op)
            x_value = sess.run(x)
            print(demo_name, "迭代%s次后，x为%f" % (i, x_value))
# end def


def train_exp_decay(demo_name, training_times):
    """
    指数衰减法，自动调整训练速率
    :param demo_name: 训练名称
    :param training_times: 训练次数
    :return:
    """
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.1, global_step, 1, 0.96, staircase=True)
    x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
    loss = tf.square(x)
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(1, 1 + training_times):
            sess.run(train_op)
            learning_rate_value = sess.run(learning_rate)
            x_value = sess.run(x)
            print(demo_name, "迭代%s次后，x为%f，学习率为%f" % (i, x_value, learning_rate_value))


if __name__ == '__main__':
    # 学习速率过大导致震荡
    train_constant_rate("demo1", 1, 10)
    print()

    # 学习速率过小导致下降速度过慢
    train_constant_rate("demo2", 0.001, 10)
    print()

    # 指数衰减法
    train_exp_decay("demo3", 10)
