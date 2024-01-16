import tensorflow as tf
from tensorflow.keras import layers, models

# MNISTデータセットのロード
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 画像データの正規化
train_images = train_images / 255.0
test_images = test_images / 255.0

# モデルの構築
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# モデルのコンパイル
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# モデルのトレーニング
model.fit(train_images, train_labels, epochs=5)

# テストデータでの評価
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'\nTest accuracy: {test_acc}')