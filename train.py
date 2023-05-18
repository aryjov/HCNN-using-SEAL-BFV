import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras import layers, Input
from keras.layers import Dense, Activation, Flatten, Conv2D, Input
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def sqr2(x):
    return x * x

conv1= Conv2D(filters=5, kernel_size=(5,5), strides=(2,2), use_bias= False)
activ1 = Activation(sqr2)
conv2 =  Conv2D(filters=50, kernel_size=(5,5), strides=(2,2), use_bias= False)
activ2 = Activation(sqr2)
dense1 = Dense(10, use_bias = False)



(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train / 255 
x_test = tf.math.round((x_test /255) * 4)


y_train = to_categorical(y_train, 10) 
y_test_cat = to_categorical(y_test, 10)



inputs = Input(shape = (28, 28, 1), name="img")

conv1.set_weights(tf.math.round(np.array(tf.multiply(15.0, conv1.get_weights()))) / 15.0)
y = conv1(inputs)
y = activ1(y)
conv2.set_weights(tf.math.round(np.array(tf.multiply(15.0, conv2.get_weights()))) / 15.0)
y = conv2(y)
y = activ2(y)
y= Flatten()(y)
dense1.set_weights(tf.math.round(np.array(tf.multiply(15.0, dense1.get_weights()))) / 15.0)
y = dense1(y)
outputs = y



model = keras.Model(inputs, outputs, name="HCNN")



model.compile(optimizer =tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=10)
model.layers[1].set_weights(tf.math.round(np.array(tf.multiply(15.0, model.layers[1].get_weights()))) / 15.0)
model.layers[3].set_weights(tf.math.round(np.array(tf.multiply(15.0, model.layers[3].get_weights()))) / 15.0)
model.layers[6].set_weights(tf.math.round(np.array(tf.multiply(15.0, model.layers[6].get_weights()))) / 15.0)

model.layers[1].set_weights(tf.math.round(tf.math.multiply(model.layers[1].get_weights(), 15)))
model.layers[3].set_weights(tf.math.round(tf.math.multiply(model.layers[3].get_weights(), 15)))
model.layers[6].set_weights(tf.math.round(tf.math.multiply(model.layers[6].get_weights(), 15)))


print(model.evaluate(x_test, y_test_cat) )

model.save('HCNN')
