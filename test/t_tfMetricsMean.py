import tensorflow as tf

# Meanメトリクスインスタンスの作成
mean = tf.keras.metrics.Mean()

# データの追加
mean.update_state(5)
# mean.update_state(10)
mean.update_state(10, sample_weight=2)
# 現在の平均値の取得
current_mean = mean.result().numpy()
print("Current mean:", current_mean) #8.33333...

# メトリクスのリセット
mean.reset_states()