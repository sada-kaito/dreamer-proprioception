def my_function(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# 関数呼び出し時に任意のキーワード引数を渡す
my_function(arg1='Hello', arg2='World', arg3=123, a='kk')

my_dict = dict(key1='value1', key2='value2', key3='value2')
print(my_dict['key1'])
a = 1
print(f'{a}は1だ')