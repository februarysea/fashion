# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


# Download dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),       # 扁平化 将二维数组转化为一维数组
    keras.layers.Dense(128, activation=tf.nn.relu),   # 128个节点
    keras.layers.Dense(10, activation=tf.nn.softmax)  # 返回10个概率得分的数组，总和为1
])

model.compile(
    optimizer=tf.train.AdamOptimizer(),      # 损失函数 衡量准确率 尽可能小
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])                    # 显示准确率

model.fit(train_images, train_labels, epochs=5)  # 拟合


test_loss, test_acc = model.evaluate(test_images, test_labels)

predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))