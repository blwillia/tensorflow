import tensorflow as tf
import numpy as np
from tensorflow import keras
# tf.test.is_gpu_available()
# tf.test.gpu_device_name()

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

with tf.device('/gpu:0'):
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=float)
    ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0], dtype=float)

model.fit(xs, ys, epochs=50)

print(model.predict([10.0]))

