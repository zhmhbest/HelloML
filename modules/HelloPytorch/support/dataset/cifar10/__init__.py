
def load_data():
    import os
    from torchvision.datasets import CIFAR10
    __cd__ = os.path.dirname(__file__)
    dataset_train = CIFAR10(root=__cd__, train=True, download=True)
    dataset_test = CIFAR10(root=__cd__, train=False, download=True)
    return (
        dataset_train.data,
        dataset_train.targets,
        dataset_test.data,
        dataset_test.targets,
        dataset_train.classes
    )
    # x_train, y_train, x_test, y_test, y_classes = load_data()


def show_images(x_data, y_data, y_classes):
    import threading
    import numpy as np
    from matplotlib import pyplot as plt

    class SubThread (threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)

        def run(self):
            print([y_classes[_i_] for _i_ in y_data])
            plt.imshow(np.hstack(x_data))
            plt.show()

    # 启动线程
    SubThread().start()
    # show_images(x_train[0:10], y_train[0:10], y_classes)
