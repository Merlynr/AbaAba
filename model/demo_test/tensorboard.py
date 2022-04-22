# %% TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import tensorflow as tf
import datetime
import os

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Model

# import tensorflow_datasets as tfds
#%%
# 制作数据集

# def load_dataset():
#     mnist = tf.keras.datasets.mnist
#     return mnist['x_train'] / 255.0, mnist['y_train'], mnist['x_test'] / 255.0, mnist['y_test']


def load_dataset():
    mnist = np.load("datasets/mnist.npz")
    return mnist['x_train'] / 255.0, mnist['y_train'], mnist['x_test'] / 255.0, mnist['y_test']


x_train, y_train, x_test, y_test = load_dataset()

print(np.shape(x_train))

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# y_train = tf.one_hot(y_train, depth=10)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
#%%
class MyModel(Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes

        self.conv1 = Conv2D(2, 3, activation='relu')
        self.pool1 = MaxPool2D(pool_size=(2, 2))
        self.conv2 = Conv2D(4, 3, activation='relu')
        self.pool2 = MaxPool2D(pool_size=(2, 2))
        self.flatten = Flatten()
        self.d1 = Dense(28, activation='relu')
        self.d2 = Dense(10)

    @tf.function(input_signature=[tf.TensorSpec([None, 28, 28, 1], tf.float32, name='inputs')])
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
#%%
def training_model():
    model = MyModel()
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_obj(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)  # 反向求导
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # 参数更新

        train_loss(loss)
        train_acc(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_obj(labels, predictions)

        test_loss(t_loss)
        test_acc(labels, predictions)

    stamp = datetime.datetime.now().strftime("Y%m%d-%H%M%S")
    logdir = os.path.join("logs/" + stamp)
    summary_writer = tf.summary.create_file_writer(logdir)

    tf.summary.trace_on(graph=True, profiler=False)  # 开启trace，并记录图结构和profile信息

    Epochs = 10

    for epoch in range(Epochs):
        # 在下一次epoch开始时，重置评估指标
        train_loss.reset_states()
        train_acc.reset_states()
        test_loss.reset_states()
        test_acc.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        with summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuarcy', train_acc.result(), step=epoch)

        for images, labels in test_ds:
            test_step(images, labels)

        with summary_writer.as_default():
            tf.summary.scalar('test_loss', test_loss.result(), step=epoch)
            tf.summary.scalar('test_accuarcy', test_acc.result(), step=epoch)

        template = 'Epoch {}, Loss: {}, Acc: {}, TestAcc: {}'
        print(template.format(epoch + 1, train_loss.result(), train_acc.result() * 100, test_acc.result() * 100))
    with summary_writer.as_default():
        tf.summary.trace_export(name="model_trace", step=3, profiler_outdir=None)
#%%
#tensorboard --logdir=logs
training_model()