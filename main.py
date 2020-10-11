import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0],-1)
x_test = x_test.reshape(x_test.shape[0],-1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

num_labels = len(np.unique(y_train))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = models.Sequential()
model.add(Dense(256, input_dim = 28*28))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=20, batch_size = 128)

loss, acc = model.evaluate(x_test, y_test, batch_size = 128)
print("Test accuracy: %.1f%%" % (100.0 * acc))

import matplotlib.pyplot as plt

plt.figure(figsize = (13,7))
plt.plot(history.history['accuracy'])
plt.title('model accuracy', fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.ylabel('accuracy', fontsize = 15)
plt.xlabel('epoch', fontsize = 15)
plt.show()