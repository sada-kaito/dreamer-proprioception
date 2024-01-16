import tensorflow as tf

y = tf.constant([-8, 5, 0.6, -0.3])
y = tf.cast(y, tf.float64)
# y = tf.less_equal(tf.abs(y), 1.)
# print(y)

y = tf.clip_by_value(y, -0.99999997, 0.99999997)
print(y)
# tf.Tensor([-0.99999994  0.99999994  0.6  -0.3 ], shape=(4,), dtype=float32)

y = tf.clip_by_value(y, -0.99999998, 0.99999998)
print(y)
# tf.Tensor([-1.   1.   0.6 -0.3], shape=(4,), dtype=float32)

y = tf.clip_by_value(y, -0.995, 0.995)
print(y)
# tf.Tensor([-0.995  0.995  0.6   -0.3  ], shape=(4,), dtype=float32)


# なぜか答えが変わるし，0.99999997でクリップされてない
# float64で計算しなおしたら，上記の答えが下記のようになった．
# tf.Tensor([-0.99999997  0.99999997  0.60000002 -0.30000001], shape=(4,), dtype=float64)
# tf.Tensor([-0.99999997  0.99999997  0.60000002 -0.30000001], shape=(4,), dtype=float64)
# tf.Tensor([-0.995       0.995       0.60000002 -0.30000001], shape=(4,), dtype=float64)

# このような結果になるのはおそらく計算過程での数値丸めで起きると思われる．