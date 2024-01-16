import numpy as np
from dm_control import suite
# import tensorflow as tf
# from tensorflow import keras
# from keras import layers
import matplotlib.pyplot as plt

# 環境を選択します。ここでは 'cartpole' の 'swingup' タスクを使用します。
# env = suite.load(domain_name="pendulum", task_name="swingup")
env = suite.load(domain_name="walker", task_name="walk")
# 環境を初期化し、最初の観測を取得します。
time_step = env.reset()
print(time_step)
plt.imshow(env.physics.render())

# 環境内でランダムなアクションを100ステップ実行します。
for _ in range(100):
    # action = np.random.uniform(low=-2.0, high=2.0, size=env.action_spec().shape)
    # if _ % 2==0 or _ > 70:
    #     action = np.zeros(6)
    action = [0, 0, 0,     # right [腰，膝，足
              0.3, 0, -0]   # left   腰，膝，足]
    time_step = env.step(action)
    
    # else:
    #     time_step = env.step(-2)
    # print(time_step)
    # print(f"orientations:  {time_step[3]['orientations'][0:2]}")
    # print()
    # print(f"right: {time_step[3]['orientations'][2:8]}")
    # print()
    # print(f"left:  {time_step[3]['orientations'][8:14]}")
    # print()
    # print(f"height:  {time_step[3]['height']}")
    # print()
    # print(f"velocity:  {time_step[3]['velocity']}")
    # print()
    # 現在の状態を描画します。
    plt.axis('off')
    plt.imshow(env.physics.render(camera_id='side'))
    plt.pause(0.001)  # 少し待機して画像を見えるようにします。
    # print(env.physics.data.time)

