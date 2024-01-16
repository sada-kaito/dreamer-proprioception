import tensorflow as tf
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent))
import models

obs = {'image': tf.random.uniform((5, 64, 64, 3), minval=0, maxval=1)}
# print(obs['image'].shape)
# print(tf.shape(obs['image']).numpy())
x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
print(tuple(obs['image'].shape[-3:]))
encoder = models.ConvEncoder()

x = encoder(x)
# print(x.shape)
depth = 32
shape = tf.concat([tf.shape(obs['image'])[:-3], [32 * depth]], 0)
batch_size = tf.shape(obs['image'])[:-3]
# print(batch_size)
x_reshape = tf.reshape(x, shape)


# Encoderの出力　(batch_size, hight, width, channels)
# 一列の特徴量 (bathc_size, feature)
# (5,2,2,256)->　(5,1024)
