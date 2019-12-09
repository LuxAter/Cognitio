#!/usr/bin/env python3

from time import time

from tensorflow import keras
import numpy as np


def gen_data(count):
    train_x = np.asarray([np.random.uniform(-10, 10, 2) for i in range(count)])
    train_y = np.asarray([x[0]**3-3*x[0]*x[1]**2 for x in train_x])
    test_x= np.asarray([np.random.uniform(-10, 10, 2) for i in range(count)])
    test_y= np.asarray([x[0]**3-3*x[0]*x[1]**2 for x in train_x])
    return train_x, train_y, test_x, test_y


def main():
    train_x, train_y, test_x, test_y = gen_data(50000)
    model = keras.Sequential([
        keras.layers.Dense(512, input_shape=(2,), activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(1, activation='relu')
        ])
    model.compile(optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_squared_error'])
    file_path = ".{}_log/{}".format(__file__.strip("./").strip(".py"), time())
    tensorboard = keras.callbacks.TensorBoard(log_dir=file_path)
    data_save = keras.callbacks.CSVLogger('{}/log.csv'.format(file_path),
            append=True,
            separator=',')
    model_save = keras.callbacks.ModelCheckpoint(
            '{}/{{epoch:05}}.h5'.format(file_path), period=10)
    model.summary()
    model.fit(train_x,
            train_y,
            epochs=100,
            callbacks=[tensorboard, model_save, data_save],
            validation_data=(test_x, test_y))


if __name__ == "__main__":
    main()
