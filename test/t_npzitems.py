import numpy as np
import matplotlib.pyplot as plt
import time

# ファイルパスを指定します
file_path = './episodes/20231210T055355-a4af09d71e5a48cda183886659a11dd0-501.npz'

data = np.load(file_path)

for k, v in data.items():
    print(k)
    print(v.shape)

orien = data['orientations']
# print(data['orientations'])
discount = data['discount']
height = data['height']
velocity = data['velocity']
action = data['action']
reward = data['reward']
