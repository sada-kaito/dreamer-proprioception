import numpy as np
import matplotlib.pyplot as plt
import time

# ファイルパスを指定します
file_path = './episodes/20231207T173639-7869aab710ca446c959720c1cd520858-501.npz'

# ファイルを読み込みます
data = np.load(file_path)
for k, v in data.items():
    print(k)
    
images = data['image']
reward = data['reward']
print(reward)
# for key in data:
#     print(f'{key}')
#     print(f'{data[key].shape}')
    
# print(f'{data["orientations"]}')
t = 0
def show_images(images, pause_time=0.02):
    # global t
    for image in images:
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        time.sleep(pause_time)  # 画像表示の間隔（秒）
        # t += pause_time
        # print(t)


start_time = time.time()

# 関数を呼び出して画像を表示
show_images(images)

end_time = time.time()
execution_time = end_time - start_time
print(f"実行時間:　{execution_time}")

# print(data['action'])