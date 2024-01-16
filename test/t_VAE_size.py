import tensorflow as tf
from tensorflow.keras import layers as kl
from tensorflow_probability import distributions as tfd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import time

class ConvEncoder(tf.keras.Model):
    
    def __init__(self, depth=3, act=tf.nn.relu):
        super().__init__()
        self.h1 = kl.Conv2D(1 * depth, 4, activation=act, strides=2)
        self.h2 = kl.Conv2D(2 * depth, 4, activation=act, strides=2)
        self.h3 = kl.Conv2D(4 * depth, 4, activation=act, strides=2)
        self.h_mean = kl.Dense(12, activation=None)
        self.h_std = kl.Dense(12, activation=None)

    # @tf.function
    def call(self, x):
        x = self.h1(x) #入力shape: (1, 28, 28, 1) 出力shape: (1, 31, 31, 32)
        print(f"en_h1:{x.shape}")
        x = self.h2(x) #出力shape: (1, 14, 14, 64)
        print(f"en_h2:{x.shape}")
        x = self.h3(x) #出力shape: (1, 1, 1, 128)
        print(f"en_h3:{x.shape}")
        mean = self.h_mean(x)
        std = self.h_std(x)
        std = tf.nn.softplus(std) + 0.1
        dist = tfd.Normal(mean, std)
        x = dist.sample()
        return x


class ConvDecoder(tf.keras.Model):
    
    def __init__(self, depth=3, act=tf.nn.relu, shape=(28, 28, 1)):
        super().__init__()
        self._depth = depth
        self._shape = shape
        self.h1_de = kl.Dense(4 * depth, activation = None)
        self.h2_de = kl.Conv2DTranspose(4 * depth, 4, activation=act, strides=2)
        self.h3_de = kl.Conv2DTranspose(2 * depth, 4, activation=act, strides=2)
        self.h4_de = kl.Conv2DTranspose(1 * depth, 6, activation=act, strides=2)
        self.h5_de = kl.Conv2DTranspose(shape[-1], 6, activation=None,strides=2)
    
    # @tf.function
    def call(self, features):
        x = self.h1_de(features)
        print(f"de_h1:{x.shape}")
        x = tf.reshape(x, [-1, 1, 1, 4 * self._depth])
        # Conv2D系は4次元の入力を想定しているためreshape(batch,1024)->(batch,1,1,1024)
        # x = self.h2_de(x) # 出力shape: (batch, 5, 5, 128)
        x = self.h3_de(x) # 出力shape: (batch, 13, 13, 64)
        print(f"de_h3:{x.shape}")
        x = self.h4_de(x) # 出力shape: (batch, 30, 30, 32)
        print(f"de_h4:{x.shape}")
        x = self.h5_de(x) # 出力shape: (batch, 64, 64, 3)
        print(f"de_h5:{x.shape}")
        shape = tf.concat([tf.shape(features)[:-1], self._shape], 0)
        mean = tf.reshape(x, shape)
        # print(f'mean_shape = {mean.shape}')
        return  tfd.Independent(tfd.Normal(mean, 1), len(self._shape))


# class VAE(tf.keras.Model):
    
#     def __init__(self):
#         super().__init__()
#         self.encoder = ConvEncoder()
#         self.decoder = ConvDecoder()
#         self.opt = tf.keras.optimizers.Adam(1e-4)
    
#     @tf.function
#     def train(self, data):
#         with tf.GradientTape() as vae_tape:
#             embed = self.encoder(data)
#             image_pred = self.decoder(embed)
#             image_logprob = image_pred.log_prob(data)
#             loss = - tf.reduce_mean(image_logprob)
            
#         gradients = vae_tape.gradient(loss, self.variables)
#         # for gradient in gradients:  
#             # print(tf.norm(gradient))
        
#         self.opt.apply_gradients(zip(gradients, self.variables))
#         return loss
        
#     @tf.function
#     def sample(self, data):
#         embed = self.encoder(data)
#         embed = tf.squeeze(embed, [1, 2])
#         image_pred_dist = self.decoder(embed)
#         image_pred = image_pred_dist.sample()
#         return image_pred


encoder = ConvEncoder()
decoder = ConvDecoder()

figure = tf.random.uniform(shape=(1, 28, 28, 1))

embed = encoder(figure)
print(embed.shape)
recon_fig = decoder(embed)


