import tensorflow as tf
from tensorflow.keras import layers as kl
import matplotlib.pyplot as plt
import numpy as np


class ConvEncoder(tf.keras.Model):
    
    def __init__(self, depth=32, act=tf.nn.relu):
        super().__init__()
        self.h1 = kl.Conv2D(1 * depth, 4, activation=act, strides=2)
        self.h2 = kl.Conv2D(2 * depth, 4, activation=act, strides=2)
        self.h3 = kl.Conv2D(4 * depth, 4, activation=act, strides=2)
        self.h4 = kl.Conv2D(8 * depth, 4, activation=act, strides=2)
        
    def call(self, x):
        x = self.h1(x) #入力shape: (1, 64, 64, 1) 出力shape: (1, 31, 31, 32)
        x = self.h2(x) #出力shape: (1, 14, 14, 64)
        x = self.h3(x) #出力shape: (1, 6, 6, 96)
        x = self.h4(x) #出力shape: (1, 2, 2, 128)
        return x

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


encoder = ConvEncoder()
train_images = train_images.reshape(60000, 28, 28, 1)
train_images = train_images.astype("float32") / 255
# feature = encoder(train_images)


#　画像を描画
image = train_images[1]
label = train_labels[1]
# print(image[15, 8])
# 画像の一部を変更
image[15, 8] = 2
# image = np.random.random((8, 8, 3))
# image[:, :, 0] = np.ones((8, 8))
# image = np.random.random((2048, 2048, 3))
plt.imshow(image, cmap='gray')
plt.title(f"Label: {label}")
plt.axis('off')
plt.show()

# CIFAR-10：　データ数5万の画像データ(解像度:32*32)
# cifar = tf.keras.datasets.cifar10
# (train_images_c, train_labels_c), (test_images_c, test_labels_c) = cifar.load_data()

# image = train_images_c[1]
# label = train_labels_c[1]

# plt.imshow(image, cmap='gray')
# plt.title(f"Label: {label}")
# plt.axis('off')
# plt.show()

# ConvEncodertest
figure = tf.random.uniform((3, 64, 64, 3), minval=0, maxval=1)
# Conv2Dへの入力shape(batch_size, height, width, channels)
print(figure.shape)  # 出力: (1, 64, 64)
# print(figure.dtype)  # 出力: <dtype: 'float32'>
# fig_numpy = figure.numpy()
# print(fig_numpy)
encoder = ConvEncoder()
x = encoder(figure)
print(x.shape)
depth = 32

shape = tf.concat([tf.shape(figure)[:-3], [32 * depth]], 0)
print(shape)