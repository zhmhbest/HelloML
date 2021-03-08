from attention.common_header import *


def test_data_generator(
        num_batches: int,
        batch_size: int,
        time_step: int,
        feature_size: int,
        target_size: int = 1
) -> (Tensor, Tensor):
    w = torch.randn(feature_size, target_size)
    for i in range(num_batches):
        x_batch = Variable(torch.randn(
            batch_size, time_step, feature_size), requires_grad=False)
        # y_batch = Variable(torch.randn(
        #     batch_size, time_step, target_size), requires_grad=False)
        y_batch = Variable(torch.matmul(x_batch, w), requires_grad=False)
        yield x_batch, y_batch


def get_test_data(**kwargs):
    buffer = []
    for xy_batch in test_data_generator(**kwargs):
        buffer.append(xy_batch)
    return buffer


if __name__ == '__main__':
    for _x, _y in test_data_generator(1, 10, 5, 8, 3):
        print(_x.shape, _y.shape)

