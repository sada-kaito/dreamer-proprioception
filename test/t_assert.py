a = 22
my_dict = dict(key1='value1', key2=222)

assert a==my_dict['key2'], 'aの値はmy_dictのkey2の値[{0}]とは異なる'.format(my_dict['key2'])
print('{0}'.format(a))