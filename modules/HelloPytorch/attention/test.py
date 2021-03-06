from attention.common_header import *


def generate_test_data(num_batches: int, batch_size: int, feature_size, target_size: int = 1, factor: int = 11):
    for i in range(num_batches):
        x_data = torch.randn(batch_size, feature_size)
        y_data = torch.randn(batch_size, target_size)
        x_batch = Variable(x_data, requires_grad=False)
        y_batch = Variable(y_data, requires_grad=False)
        yield x_batch, y_batch


if __name__ == '__main__':
    for _x, _y in generate_test_data(30, 20, 10, 11):
        print(_x.shape, _y.shape)

