from typing import Tuple
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from tensorflow.python.framework import tensor_shape


class SimpleDense(Layer):
    def __init__(self, units: int):
        super(SimpleDense, self).__init__()
        self.units = units
        self.kernel = None
        self.bias = None

    def build(self, input_shape: Tuple[int, ...]):
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = input_shape[-1]
        print(last_dim)
        # 确定可训练参数
        self.kernel = self.add_weight(
            'kernel',
            shape=(last_dim, self.units),
            initializer=initializers.glorot_uniform,
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer=initializers.zeros,
            trainable=True
        )

    def call(self, inputs, **kwargs):
        # 在此处编写逻辑
        return K.bias_add(
            K.dot(inputs, self.kernel),
            self.bias
        )

    def compute_output_shape(self, input_shape: Tuple[int, ...]):
        input_shape = tensor_shape.TensorShape(input_shape)
        return input_shape[:-1].concatenate(self.units)


if __name__ == '__main__':
    import numpy as np
    x_test = np.random.randn(1, 32)
    print(x_test.shape)
    d = SimpleDense(10)
    y_pred = d(x_test)
    print(y_pred)
    print(y_pred.shape)
