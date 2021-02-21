# import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


def __is_number(arg):
    return type(arg) is float or type(arg) is int


def __init_w_b(w, b):
    """
    智能初始化
    :param w:
    :param b:
    :return:
    """
    # 权重初始方法
    w_initializer = (lambda: tf.truncated_normal_initializer(stddev=w)) if __is_number(w) else w
    # print(type(w_initializer()))

    # 偏执初始方法
    b_initializer = (lambda: tf.constant_initializer(b)) if __is_number(b) else b
    # print(type(b_initializer()))

    return w_initializer, b_initializer
