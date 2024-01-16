import tensorflow as tf

outputs = [[] for i in range(8)]
batch_size = 3
state = dict(
    mean=tf.zeros([batch_size, 2]),
    std=tf.zeros([batch_size, 3]),
    stoch=tf.zeros([batch_size, 4]),
    deter=tf.zeros([batch_size, 5]))
last = state, state

# print(tf.nest.flatten(last))
for i in range(5):
    [o.append(i) for o, i in zip(outputs, tf.nest.flatten(last))]
# print(outputs)
i = 1
for x in outputs:
    print(x)
    print(i)
    i += 1
outputs = [tf.stack(x, 0) for x in outputs]
print(outputs)

outputs = [tf.zeros([5,batch_size, 5]),
tf.zeros([5,batch_size, 2]),
tf.zeros([5,batch_size, 3]),
tf.zeros([5,batch_size, 4])]


# post, prior = tf.nest.pack_sequence_as((state, state), outputs)
prior = tf.nest.pack_sequence_as(state, outputs)
print('変換後')
print(prior)