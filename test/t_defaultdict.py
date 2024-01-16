# from collections import defaultdict

# dd_list = defaultdict(list)

# dd_list["key"].append(1)
# # dd_list["key"].append(2)
# # dd_list["key"].append(3)

# # dd_list["key"].extend([1,2])
# # dd_list["key"].extend(1)
# print(dd_list["key"])
# # print(dd_list)
# dd_list['key'].update_state(2)
# print(dd_list['key'])

import tensorflow as tf
import collections

# defaultdict で Mean インスタンスの辞書を作成
metrics = collections.defaultdict(tf.metrics.Mean)

# メトリクスの更新
metrics['accuracy'].update_state(0.9)
metrics['loss'].update_state(0.1)

# metrics['accuracy'].update_state(10)

# 結果の取得
accuracy_result = metrics['accuracy'].result().numpy()
loss_result = metrics['loss'].result().numpy()

print("Accuracy:", accuracy_result)
print("Loss:", loss_result)