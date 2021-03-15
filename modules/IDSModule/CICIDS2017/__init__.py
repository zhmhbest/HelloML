import os
import re
import pickle
import numpy as np
from . import EnhancePickle


DATA_LOCATION_DIRECTORY = f"{os.path.dirname(__file__)}"
DATA_DUMP = f"{DATA_LOCATION_DIRECTORY}/dump"
# print(DATA_LOCATION_DIRECTORY)


def get_iscx_files():
    for f in os.listdir(DATA_LOCATION_DIRECTORY):
        if re.search(r"ISCX\.csv$", f):
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


# if __name__ == '__main__':
#     import time
#     s = time.time()
#     for x_batch, y_batch in get_iscx_row_data(600):
#         pass
#         # print(x_batch.shape)
#         # print(y_batch.shape)
#         # break
#     s = time.time() - s
#     print(s)
#     # 35.31558132171631


def make_iscx_dump(batch_size: int, filename: str = DATA_DUMP):
    file_data = f"{filename}.pkl"
    file_info = f"{filename}.info"
    if os.path.exists(file_data):
        return
    writer = EnhancePickle.PickleWriteBuffer(file_data)
    num_batches = 0
    num_row = 0
    collect_label = set()
    for x_batch, y_batch in get_iscx_row_data(batch_size):
        num_row += x_batch.shape[0]
        # Batch
        x_batch = x_batch.astype(np.float64).tolist()
        y_batch['Web Attack � XSS' == y_batch] = 'Web Attack XSS'
        y_batch['Web Attack � Brute Force' == y_batch] = 'Web Attack Brute Force'
        y_batch['Web Attack � Sql Injection' == y_batch] = 'Web Attack Sql Injection'
        y_batch = y_batch.tolist()
        # Label
        for labels in y_batch:
            collect_label.add(labels[0])
        # Append
        writer.append((x_batch, y_batch))
        num_batches += 1
    writer.done()
    with open(f"{file_info}", "wb") as fp:
        collect_label = list(collect_label)
        collect_label.sort()
        obj = {
            "batch_size": batch_size,
            "num_batches": num_batches,
            "sum_data": num_row,
            "label": collect_label
        }
        pickle.dump(obj, fp)
        # print(obj)


# if __name__ == '__main__':
#     import time
#     s = time.time()
#     # ----------------------------
#     make_iscx_dump(600)
#     # ----------------------------
#     s = time.time() - s
#     print(s)


def get_iscx_dump(filename: str = DATA_DUMP):
    file_data = f"{filename}.pkl"
    file_info = f"{filename}.info"
    if not os.path.exists(file_data) or not os.path.exists(file_info):
        raise FileNotFoundError()
    with open(f"{file_info}", "rb") as fp:
        info = pickle.load(fp)
    reader = EnhancePickle.PickleReadBuffer(file_data)
    return info, reader
    # for i in range(info['num_batches']):
    #     yield reader.pop()
