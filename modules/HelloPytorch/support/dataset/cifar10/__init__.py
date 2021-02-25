
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


def load_data_fixed():
    import os
    import pickle
    import numpy as np
    dump = os.path.join(os.path.dirname(__file__), "cifar-10.pkl")
    if os.path.exists(dump):
        print("Loading cifar-10 ...")
        with open(dump, "rb") as fp:
            return pickle.load(fp)
    else:
        x_raw_train, y_raw_train, x_raw_test, y_raw_test, y_classes = load_data()
        y_raw_train = np.array(y_raw_train, dtype=np.int64)
        y_raw_test = np.array(y_raw_test, dtype=np.int64)
        with open(dump, "wb") as fp:
            pickle.dump([
                x_raw_train, y_raw_train, x_raw_test, y_raw_test, y_classes
            ], fp)
        return x_raw_train, y_raw_train, x_raw_test, y_raw_test, y_classes


def image_to_tensor_array(array_images):
    import numpy as np
    shape = array_images.shape
    buffer = []
    for i in range(shape[0]):
        one = array_images[i]
        buf = []
        for j in range(shape[-1]):
            buf.append(one[:, :, j])
        buffer.append(buf)
    return np.array(buffer) / 255


def show_images(x_data, y_data, y_classes):
    import numpy as np
    from matplotlib import pyplot as plt
    print([y_classes[_i_] for _i_ in y_data.reshape(-1).tolist()])
    plt.imshow(np.hstack(x_data))
    plt.show()

    # import threading
    # class SubThread (threading.Thread):
    #     def __init__(self):
    #         threading.Thread.__init__(self)
    #
    #     def run(self):
    #         print([y_classes[_i_] for _i_ in y_data.reshape(-1).tolist()])
    #         plt.imshow(np.hstack(x_data))
    #         plt.show()
    #
    # # 启动线程
    # SubThread().start()
    # # show_images(x_train[0:10], y_train[0:10], y_classes)
