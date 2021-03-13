from CICIDS2017 import get_iscx_row_data

for x_batch, y_batch in get_iscx_row_data(600):
    print(x_batch)
    print(x_batch.shape)
    # print(y_batch)
    print(y_batch.shape)
    break
