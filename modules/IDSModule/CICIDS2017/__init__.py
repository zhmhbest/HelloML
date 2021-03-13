import os
import re
import numpy as np
DATA_LOCATION_DIRECTORY = f"{os.path.dirname(__file__)}"
DATA_DUMP = f"{DATA_LOCATION_DIRECTORY}/dump.pkl"
# print(DATA_LOCATION_DIRECTORY)


def get_iscx_files():
    for f in os.listdir(DATA_LOCATION_DIRECTORY):
        if re.search("ISCX\.csv$", f):
            yield f


def get_iscx_row_data(batch_size: int, exclude_head: bool = True) -> (np.ndarray, np.ndarray):
    """
    按行获取CSV生数据
    :param batch_size: 每批大小
    :param exclude_head: 跳过Head
    :return:
    """
    sum_line: int = 0
    read_buffer = []
    for f in get_iscx_files():
        print("Loading", f)
        # print("Sum, Remain =", sum_line, sum_line % batch_size)
        with open(f"{DATA_LOCATION_DIRECTORY}/{f}", "r", encoding="utf-8") as fp:
            if exclude_head:
                fp.readline()
            while True:
                line = fp.readline()
                if not line:
                    break
                line = line.strip()
                if 0 == len(line):
                    break
                read_buffer.append(line.split(","))
                sum_line += 1
                if 0 == sum_line % batch_size:
                    xy_batch = np.array(read_buffer)
                    # print(xy_batch.shape)
                    yield xy_batch[:, 0:-1], xy_batch[:, -1:]
                    read_buffer.clear()
    # final
    if sum_line % batch_size > 0:
        xy_batch = np.array(read_buffer)
        yield xy_batch[:, 0:-1], xy_batch[:, -1:]
    print("Sum:", sum_line)
    print("=== This is the end of row data fetch ===")


if __name__ == '__main__':
    import time
    s = time.time()
    for x_batch, y_batch in get_iscx_row_data(600):
        pass
        # print(x_batch.shape)
        # print(y_batch.shape)
        # break
    s = time.time() - s
    print(s)
    # 35.31558132171631
