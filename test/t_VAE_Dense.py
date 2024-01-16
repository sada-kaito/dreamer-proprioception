import tensorflow as tf
from tensorflow.keras import layers as kl
from tensorflow_probability import distributions as tfd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import time

class Encoder(tf.keras.Model):
    
    def __init__(self, act=tf.nn.relu):
        super().__init__()
        self.h1 = kl.Dense(256, activation=act)
        self.h2 = kl.Dense(128, activation=act)
        self.h3 = kl.Dense(64, activation=act)
        self.h_mean = kl.Dense(5, activation=None)
        self.h_std = kl.Dense(5, activation=None)

    # @tf.function
    def call(self, x):
        x = tf.reshape(x, [x.shape[0], -1])
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
    
class Decoder(tf.keras.Model):
    
    def __init__(self, act=tf.nn.relu, shape=(28, 28)):
        super().__init__()
        self.shape = shape
        self.h1 = kl.Dense(64, activation=act)
        self.h2 = kl.Dense(128, activation=act)
        self.h3 = kl.Dense(256, activation=act)
        self.h_mean = kl.Dense(784, activation=None)

    # @tf.function
    def call(self, x):
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        mean = self.h_mean(x)
        shape = tf.concat([tf.shape(x)[:-1], self.shape], 0)
        mean = tf.reshape(mean, shape)
        # print(mean.shape)
        return  tfd.Independent(tfd.Normal(mean, 0.000001), len(self.shape))
    

class VAE(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.opt = tf.keras.optimizers.Adam(5e-3)
    
    def train(self, data, steps):
        with tf.GradientTape() as vae_tape:
            embed = self.encoder(data)
            image_pred = self.decoder(embed)
            image_logprob = image_pred.log_prob(data)
            loss = - tf.reduce_mean(image_logprob)
            
        gradients = vae_tape.gradient(loss, self.variables)

        self.opt.apply_gradients(zip(gradients, self.variables))
        return loss
        
    def sample(self, data):
        embed = self.encoder(data)
        image_pred_dist = self.decoder(embed)
        image_pred = image_pred_dist.sample()
        return image_pred


def generate_and_images(model, epoch, test_sample, labels):
    predictions = model.sample(test_sample)
    
    plt.figure(figsize=(6.4, 6.8))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.title(f"Label: {labels[i]}")
        plt.imshow(test_sample[i, :, :], cmap='gray')
        plt.axis('off')
    plt.show()
    
    plt.figure(figsize=(6.4, 6.8))
    for i in range(16):
        
        plt.subplot(4, 4, i + 1)
        plt.title(f"Label: {labels[i]}")
        plt.imshow(predictions[i, :, :], cmap='gray')
        plt.axis('off')
    plt.show()


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# image_logprob = vae()
# print(train_image)
# train_images = tf.expand_dims(train_images, -1)
# test_images = tf.expand_dims(test_images, -1)

train_images = tf.cast(train_images, tf.float32) / 255.0
test_images = tf.cast(test_images, tf.float32) / 255.0


vae = VAE()

# データセットのbatch化
train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(10000).batch(256))

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

