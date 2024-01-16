import tensorflow as tf
from tensorflow.keras import layers as kl
from tensorflow_probability import distributions as tfd

class ConvDecoder(tf.keras.Model):
    
    def __init__(self, depth=32, act=tf.nn.relu, shape=(64, 64, 3)):
        super().__init__()
        self._depth = depth
        self._shape = shape
        self.h1_de = kl.Dense(32 * depth, activation = None)
        self.h2_de = kl.Conv2DTranspose(4 * depth, 5, activation=act, strides=2)
        self.h3_de = kl.Conv2DTranspose(2 * depth, 5, activation=act, strides=2)
        self.h4_de = kl.Conv2DTranspose(1 * depth, 6, activation=act, strides=2)
        self.h5_de = kl.Conv2DTranspose(shape[-1], 6, activation=None,strides=2)
        
    def call(self, features):
        x = self.h1_de(features)
        x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
        # Conv2D系は4次元の入力を想定しているためreshape(batch,1024)->(batch,1,1,1024)
        x = self.h2_de(x) # 出力shape: (batch, 5, 5, 128)
        x = self.h3_de(x) # 出力shape: (batch, 13, 13, 64)
        x = self.h4_de(x) # 出力shape: (batch, 30, 30, 32)
        x = self.h5_de(x) # 出力shape: (batch, 64, 64, 3)
        shape = tf.concat([tf.shape(features)[:-1], self._shape], 0)
        print(tf.shape(features)[0])
        print(tf.shape(features)[-1])
        print(tf.shape(features)[:-1])
        print(shape)
        mean = tf.reshape(x, shape)
        return tfd.Independent(tfd.Normal(mean, 1), len(self._shape)), shape, x.shape
    
    
decoder = ConvDecoder()

features = tf.random.uniform((5, 1024), minval=0, maxval=1)
x, shape, x_shape = decoder(features)

# print(x.sample().shape)
# print(shape)
# print(x_shape)

shape = (64, 64, 3)
# print(len(shape))