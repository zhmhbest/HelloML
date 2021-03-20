import pickle
import numpy as np
from sklearn.preprocessing import Normalizer, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD

"""
    数据载入
"""
with open("./data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.pkl", "rb") as fd:
    xy_data = np.array(pickle.load(fd))
    x_data = xy_data[:, :78]
    y_data = xy_data[:, 78:]


"""
    数据预处理
"""
norm = StandardScaler()
x_data = norm.fit_transform(x_data)

one = OneHotEncoder(categories='auto')
y_data = one.fit_transform(y_data).toarray()


"""
    数据划分
"""
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)


"""
    模型
"""
model = Sequential()
model.add(Dense(8, input_dim=x_data.shape[1]))
model.add(Activation('tanh'))
model.add(Dense(y_data.shape[1]))
model.compile(
    loss=categorical_crossentropy,
    optimizer=SGD(lr=0.001)
)


"""
    模型训练（首次）
"""
model.fit(
    x_train, y_train,  # 训练数据
    batch_size=32, epochs=10
)


"""
    模型评估
"""
score = model.evaluate(x_test, y_test, batch_size=32, verbose=0)
print("score =", score)
