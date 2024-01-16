from tensorflow_probability import distributions as tfd
import tensorflow as tf

batch_size = 100
shape = (64, 64, 3)

mean = tf.random.uniform((batch_size, 64, 64, 3), minval=0, maxval=1)
normal_dist = tfd.Normal(mean, scale=1)

print(normal_dist.sample().shape) # (100, 64, 64, 3)

dist = tfd.Independent(normal_dist, reinterpreted_batch_ndims=len(shape))
print(len(shape)) # 3
print(dist)
# 4次元の値をbatch_shapeとevent_shapeに分ける
# tfp.distributions.Independent("IndependentNormal", 
#                               batch_shape=[100], 
#                               event_shape=[64, 64, 3], 
#                               dtype=float32)
print(normal_dist[0].sample().shape) # (64, 64, 3)
print(dist[0].sample().shape) # (64, 64, 3)

# print(dist.sample(10))
# 答えは一緒である．わざわざ分ける利点については現時点では不明

normal_dist_logprob = normal_dist.log_prob(mean)
dist_logprob = normal_dist.log_prob(mean)
print(normal_dist_logprob.shape)
print(dist_logprob.shape)

#追記：logprobを計算するときに必要になる．