import tensorflow as tf
import tensorflow_probability as tfp

y = tf.constant([-8, 5, 0.6, -0.3])
y = tf.cast(y, tf.float32)
bool_list = tf.less_equal(tf.abs(y), 1.)

# 論文での実装
y = tf.where(tf.less_equal(tf.abs(y), 1.),
             tf.clip_by_value(y, -0.99999997, 0.99999997), y)
print(bool_list)
# tf.Tensor([False False  True  True], shape=(4,), dtype=bool)
print(y)
# tf.Tensor([-8.   5.   0.6 -0.3], shape=(4,), dtype=float32)

#　計算したyをarctanhに入力するならば,多分以下が正しいのではないか
y = tf.where(tf.less_equal(tf.abs(y), 1.),
             y, tf.clip_by_value(y, -0.99999997, 0.99999997))

print(bool_list)
# tf.Tensor([False False  True  True], shape=(4,), dtype=bool)
print(y)
# tf.Tensor([-0.99999994  0.99999994  0.6  -0.3 ], shape=(4,), dtype=float32)

y = tf.constant([-8, 5, 0.6, -0.3])
y = tf.cast(y, tf.float32)
y = tf.atanh(y)
print(y)
# tf.Tensor([ nan  nan  0.69314724 -0.30951962], shape=(4,), dtype=float32)