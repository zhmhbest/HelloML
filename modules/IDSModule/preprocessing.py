import os
import numpy as np
import pandas as pd
import pickle


def make_pickle(csv_name):
    csv_file_name = f"./data/CICIDS2017/{csv_name}.csv"
    csv_dump_name = f"./data/{csv_name}.pkl"
    if os.path.exists(csv_dump_name):
        return
    data = pd.read_csv(
        csv_file_name,
        header=0,
        names=pd.read_csv("./data/CICIDS2017/names.csv", squeeze=True).columns
    )
    # 修正乱码
    data['Label'] = data['Label'].apply(lambda label: str(label).replace('� ', ''))
    # Label注解
    data = data.merge(
        right=pd.read_csv("./data/CICIDS2017/labels.csv", header=0),
        on='Label',
        how='inner'
    )
    # Break
    del data['Label']

    with open(csv_dump_name, "wb") as fd:
        pickle.dump(data.to_numpy(dtype=np.int).tolist(), fd)


if __name__ == '__main__':
    make_pickle("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX")


# if __name__ == '__main__':
#     import re
#     for f in os.listdir("./data/CICIDS2017"):
#         ff = re.match(r"^(.+ISCX)\.csv$", f)
#         if ff is not None:
#             name = ff.groups()[0]
#             print(name)
#             make_pickle(name)
