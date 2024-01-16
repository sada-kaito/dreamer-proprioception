import tensorflow as tf
from tensorflow.keras import layers as kl
from tensorflow_probability import distributions as tfd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import time

class ConvEncoder(tf.keras.Model):
    
    def __init__(self, depth=8, act=tf.nn.relu):
        super().__init__()
        self.h1 = kl.Conv2D(1 * depth, 4, activation=act, strides=2)
        self.h2 = kl.Conv2D(2 * depth, 4, activation=act, strides=2)
        self.h3 = kl.Conv2D(4 * depth, 4, activation=act, strides=2)
        self.h_mean = kl.Dense(12, activation=None)
        self.h_std = kl.Dense(12, activation=None)

    # @tf.function
    def call(self, x):
        x = self.h1(x) #入力shape: (1, 28, 28, 1) 出力shape: (1, 31, 31, 32)
        # print(f"en_h1:{x.shape}")
        x = self.h2(x) #出力shape: (1, 14, 14, 64)
        # print(f"en_h2:{x.shape}")
        x = self.h3(x) #出力shape: (1, 1, 1, 128)
        # print(f"en_h3:{x.shape}")
        mean = self.h_mean(x)
        std = self.h_std(x)
        std = tf.nn.softplus(std)
        dist = tfd.Normal(mean, std)
        x = dist.sample()
        return x


class ConvDecoder(tf.keras.Model):
    
    def __init__(self, depth=8, act=tf.nn.relu, shape=(28, 28, 1)):
        super().__init__()
        self._depth = depth
        self._shape = shape
        self.h1_de = kl.Dense(4 * depth, activation = None)
        self.h2_de = kl.Conv2DTranspose(4 * depth, 4, activation=act, strides=2)
        self.h3_de = kl.Conv2DTranspose(2 * depth, 4, activation=act, strides=2)
        self.h4_de = kl.Conv2DTranspose(1 * depth, 6, activation=act, strides=2)
        self.h5_de = kl.Conv2DTranspose(shape[-1], 6, activation=None,strides=2)
    
    @tf.function
    def call(self, features):
        x = self.h1_de(features)
        x = tf.reshape(x, [-1, 1, 1, 4 * self._depth])
        # Conv2D系は4次元の入力を想定しているためreshape(batch,1024)->(batch,1,1,1024)
        # x = self.h2_de(x) # 出力shape: (batch, 5, 5, 128)
        x = self.h3_de(x) # 出力shape: (batch, 13, 13, 64)
        x = self.h4_de(x) # 出力shape: (batch, 30, 30, 32)
        x = self.h5_de(x) # 出力shape: (batch, 64, 64, 3)
        shape = tf.concat([tf.shape(features)[:-1], self._shape], 0)
        mean = tf.reshape(x, shape)
        # print(f'mean_shape = {mean.shape}')
        return  tfd.Independent(tfd.Normal(mean, 0.000001), len(self._shape))
    
class VAE(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()
        self.opt = tf.keras.optimizers.Adam(5e-3)
    
    def train(self, data, steps):
        with tf.GradientTape() as vae_tape:
            embed = self.encoder(data)
            embed = tf.squeeze(embed, [1, 2])
            image_pred = self.decoder(embed)
            image_logprob = image_pred.log_prob(data)
            loss = - tf.reduce_mean(image_logprob)
            
        gradients = vae_tape.gradient(loss, self.variables)

        self.opt.apply_gradients(zip(gradients, self.variables))
        return loss
        
    def sample(self, data):
        embed = self.encoder(data)
        embed = tf.squeeze(embed, [1, 2])
        image_pred_dist = self.decoder(embed)
        image_pred = image_pred_dist.sample()
        return image_pred


def generate_and_images(model, epoch, test_sample, labels):
    predictions = model.sample(test_sample)
    
    plt.figure(figsize=(6.4, 6.8))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.title(f"Label: {labels[i]}")
        plt.imshow(test_sample[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.show()
    
    plt.figure(figsize=(6.4, 6.8))
    for i in range(16):
        
        plt.subplot(4, 4, i + 1)
        plt.title(f"Label: {labels[i]}")
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.show()


def generate_image(test_sample):
    
    plt.imshow(test_sample[0], cmap='gray')
    plt.axis('off')
    plt.show()


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# image_logprob = vae()
# print(train_image)
train_images = tf.expand_dims(train_images, -1)
test_images = tf.expand_dims(test_images, -1)

train_images = tf.cast(train_images, tf.float32) / 255.0
test_images = tf.cast(test_images, tf.float32) / 255.0

# train_images = train_images[0:100]
# train_labels = train_labels[0:100]

# encoder = ConvEncoder()
# embed = encoder(train_images[0:100])
# print(embed.shape)
# embed = tf.squeeze(embed, [1, 2])
# print(embed.shape)

# decoder = ConvDecoder()
# dist = decoder(embed)

# print(dist.sample().shape)


## vae call test

vae = VAE()
# vae.train()
# image_logprob = vae.sample(test_images)
# image_logprob_sum = tf.reduce_mean(image_logprob)
# print(image_logprob)
# print(image_logprob_sum)


# データセットのbatch化
train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(10000).batch(256))

                                                   
## training
train_images = train_images[0:16]
train_labels = train_labels[0:16]

vae = VAE()
x = vae.sample(train_images)
epochs = 10
train_step = 1
train_steps = []
loss_list = []

for epoch in range(1, epochs+1):
    start_time = time.time()
    
    for train_data in train_dataset:
        # print(f"train_step:{train_step}")
        
        loss = vae.train(train_data, train_step)
        
        train_step +=1
        
        train_steps.append(train_step)
        loss_list.append(loss)
        # pass
    end_time = time.time()
    print(f"epoch:{epoch}, loss:{loss}")
    print(f"学習時間：{end_time-start_time}")
    generate_and_images(vae, epoch, train_images, train_labels)


plt.plot(train_steps, loss_list)
plt.xlabel("training steps")
plt.ylabel("reconstruction loss")
plt.show()
# for train_data in train_dataset:
#     generate_image(train_data)














