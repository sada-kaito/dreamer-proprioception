import tensorflow as tf

def count(stop=100):
    i = 0
    while i<stop:
        yield range(i, i+5)
        i += 1
        
dataset = tf.data.Dataset.from_generator(count, tf.int8, (5,))
dataset = dataset.batch(20, True) # batchsizeを100にするとstopiteration

dataset = iter(dataset)
data = next(dataset)
print(data)