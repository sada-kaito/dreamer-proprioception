import tensorflow_probability as tfp
import numpy as np
# 平均0、標準偏差1の正規分布
# normal_dist = tfp.distributions.Normal(loc=[0., 1, 10], scale=1)
normal_dist = tfp.distributions.Normal(loc=[[1, 2],[1, 2]], scale=1)
# 値0に対する対数確率密度
# log_prob_value = normal_dist.log_prob([1, 2, 3])
# print(log_prob_value.numpy())

print(normal_dist.sample(1))

log_prob_value = normal_dist.log_prob(normal_dist.sample([1]))
print(log_prob_value.numpy())