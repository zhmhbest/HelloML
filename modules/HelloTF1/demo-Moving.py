# import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from zhmh.dataset import BatchGenerator
from zhmh.tf import generate_network, get_l2_build_lambda, get_ema_build_lambda
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


"""
    加载数据
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
from tensorflow.python.keras.datasets.mnist import load_data
(x_train, y_train), (x_test, y_test) = load_data()

SIZE_TRAIN = x_train.shape[0]
size_test = x_test.shape[0]

x_train = x_train.reshape(SIZE_TRAIN, -1)
x_test = x_test.reshape(size_test, -1)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
print(y_train[0:10].reshape(-1))
print(y_test[0:10].reshape(-1))

one_y = OneHotEncoder()
y_train = one_y.fit_transform(y_train).toarray()
y_test = one_y.transform(y_test).toarray()

INPUT_SIZE = x_train.shape[1]
OUTPUT_SIZE = y_train.shape[1]
print(INPUT_SIZE, OUTPUT_SIZE)


"""
    创建神经网络
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
LAYER_NEURONS = [INPUT_SIZE, 500, OUTPUT_SIZE]
REGULARIZATION_RATE = 0.0001
REGULARIZER_COLLECTION = 'losses'
MOVING_AVERAGE_DECAY = 0.99
global_step = tf.Variable(0, trainable=False)

place_x = tf.placeholder(tf.float32, shape=(None, INPUT_SIZE))
place_y = tf.placeholder(tf.float32, shape=(None, OUTPUT_SIZE))

# 普通网络
y = generate_network(
    LAYER_NEURONS, place_x, 0.1, 0.001,
    get_l2_build_lambda(REGULARIZATION_RATE, REGULARIZER_COLLECTION, tf.nn.relu)
)

# 构建EMA网络
build_network_ema, ema_op = get_ema_build_lambda(global_step, MOVING_AVERAGE_DECAY, tf.nn.relu)
y_ema = generate_network(LAYER_NEURONS, place_x, 0.1, 0.001, build_network_ema, var_reuse=True)

"""
    定义损失函数
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
cross_entropy = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(place_y, 1))
)
loss = tf.add_n(tf.get_collection(REGULARIZER_COLLECTION)) + cross_entropy
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(place_y, 1)), tf.float32))
averages_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_ema, 1), tf.argmax(place_y, 1)), tf.float32))

"""
    训练
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""
LEARNING_RATE_BASE = 0.8
BATCH_SIZE = 128
LEARNING_RATE_DECAY = 0.99
TRAINING_TIMES = 3000

learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,
    global_step,
    SIZE_TRAIN / BATCH_SIZE, LEARNING_RATE_DECAY,
    staircase=True
)
batch = BatchGenerator(x_train, y_train, BATCH_SIZE, BATCH_SIZE/4)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
with tf.control_dependencies([train_op, ema_op]):
    train_ops = tf.no_op(name='train')

with tf.Session() as sess:
    # 初始化全部变量OP
    tf.global_variables_initializer().run()
    for i in range(1, 1 + TRAINING_TIMES):
        x_batch, y_batch = batch.next()
        sess.run(train_ops, feed_dict={
            place_x: x_batch,
            place_y: y_batch
        })
        if i % 500 == 0:
            va, vl, vg = sess.run([accuracy, loss, global_step], feed_dict={
                place_x: x_train,
                place_y: y_train
            })
            print(f"{vg}、accuracy={va}、loss={vl}")

    common_feed_train = {
        place_x: x_train,
        place_y: y_train
    }
    common_feed_test = {
        place_x: x_test,
        place_y: y_test
    }
    print("Train Accuracy:", accuracy.eval(common_feed_train))
    print("Test Accuracy:", accuracy.eval(common_feed_test))
    print("Train Averages Accuracy:", averages_accuracy.eval(common_feed_train))
    print("Test Averages Accuracy:", averages_accuracy.eval(common_feed_test))
    print(sess.run(
        tf.argmax(y_test[:30], 1)), "Real Number")
    print(sess.run(
        tf.argmax(y[:30], 1), feed_dict=common_feed_test), "Prediction Number")
    print(sess.run(
        tf.argmax(y_ema[:30], 1), feed_dict=common_feed_test), "Prediction Averages Number")
