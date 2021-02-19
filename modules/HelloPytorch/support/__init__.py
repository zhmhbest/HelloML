from typing import Union, Tuple


def get_cnn_filtered_size(
        input_size: Union[int, Tuple[int, int]],
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0
) -> (int, int):
    """
    :param input_size: 输入尺寸
    :param kernel_size: 过滤器尺寸
    :param stride: 过滤器步长
    :param padding: 边缘填充
    :return: 过滤后每层尺寸
    """
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    calculate = (lambda i: (input_size[i] - kernel_size[i] + 2 * padding[i]) // (stride[i]) + 1)
    return calculate(0), calculate(1)


def get_flatten_size(input_shape: Tuple[int, ...]) -> int:
    result = 1
    for num in input_shape:
        result *= num
    return result
