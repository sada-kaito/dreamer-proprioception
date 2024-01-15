import gym
import numpy as np
import matplotlib.pyplot as plt
import time

import tensorflow as tf
from tensorflow.keras.mixed_precision import global_policy
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.mixed_precision import Policy

import dreamer_repro
set_global_policy(Policy('mixed_float16'))

config = dreamer_repro.define_config()

config.batch_length = 50
config.batch_size = 1
config.train_steps = 1

datadir = config.logdir / 'episodes'
checkpointdir = config.logdir / 'checkpoints'
actspace = gym.spaces.Box(-1, 1, (6,), dtype=np.float32)
# print(config.batch_length)
agent = dreamer_repro.Dreamer(config, datadir, actspace, writer=None)
# 重みのload
agent.load()



def show_images(images, pause_time=0.04):
    # global t
    for image in images:
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        time.sleep(pause_time)  # 画像表示の間隔（秒）
        # t += pause_time
        # print(t)
        
def compare_images(real_image, reconstruction_image, steps, image_list, pause_time=0.01):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4,6))
    axes[0].imshow(real_image[0], aspect='equal')
    axes[0].set_title(f'real   step:{steps+1}')
    axes[1].imshow(reconstruction_image[0], aspect='equal')
    axes[1].set_title(f'reconstruction   step:{steps+1}')
    for ax in axes:
        ax.axis('off')
        
    plt.show()
    fig.canvas.draw()
    
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image_list.append(image)
    plt.close(fig)


dataset = iter(dreamer_repro.load_dataset(datadir, config))
obs_steps = 5

data = next(dataset)
# show_images(data_dict['image'])
embed = agent.encoder(data)
state = agent.dynamics.initial(1)
image_list = []

for i in range(obs_steps):
    post, prior = agent.dynamics.obs_step(state, data['action'][0,i:i+1], embed[0,i:i+1])
    state = post
    feat = agent.dynamics.get_feat(post)
    image_pred = agent.decoder(feat)
    reconstruction = tf.cast(image_pred.mode(), dtype=tf.float32) + 0.5
    real = tf.cast(data['image'][0,i:i+1], dtype=tf.float32) + 0.5
    compare_images(real, reconstruction, i, image_list)
    if i >= obs_steps-1:
        for k in range(config.batch_length - obs_steps):
            action = agent.actor(feat).sample()
            prior = agent.dynamics.img_step(state, action)
            state = prior
            feat = agent.dynamics.get_feat(prior)
            image_pred = agent.decoder(feat)
            reconstruction = tf.cast(image_pred.mode(), dtype=tf.float32) + 0.5
            real = tf.cast(data['image'], dtype=tf.float32)[0,i+k+1:i+k+2] + 0.5
            compare_images(real, reconstruction, i+k+1, image_list)



# from matplotlib.animation import FuncAnimation
# from PIL import Image


# fig, ax = plt.subplots(figsize=(4,6))

# def update(frame):
#     ax.clear()
#     img = Image.open(f'video/Figure 2024-01-08 144448 ({frame}).png')
#     ax.imshow(img)
#     ax.axis('off')

# ani = FuncAnimation(fig, update, frames=len(image_list), interval=200)

# ani.save('animation.gif', writer='imagemagick')


























