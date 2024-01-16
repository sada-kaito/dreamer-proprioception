import tensorflow as tf
from tensorflow.keras.mixed_precision import global_policy

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self._float = global_policy().compute_dtype
        # 他のモデルの初期化コード...

# 混合精度ポリシーを設定
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
# tf.keras.mixed_precision.set_global_policy('float32')
# モデルのインスタンス化
model = MyModel()
# モデルを作成した後にset_global_policyを使用してもmodelの中のglobal_policyは変化しない．
tf.keras.mixed_precision.set_global_policy('float16')

print(model._float)
