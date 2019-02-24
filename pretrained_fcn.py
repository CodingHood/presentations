import sys
sys.stderr = open('/dev/null', 'w')

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

import numpy as np
np.set_printoptions(precision=4, threshold=3)

### DATA PREPROCESSING ###

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))


test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

### MODEL CONSTRUCTION ###

model = models.Sequential()
model.add(layers.Dense(10, activation='relu', input_shape=(28*28,), trainable=False))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

n_rows, n_cols = 28, 28
filters = [np.zeros((n_rows, n_cols)) for i in np.arange(0, 10, 1)]

for image, label in zip(train_images[:3000], train_labels[:3000]):
    for i in range(n_rows):
        for j in range(n_cols):
            index = int(np.argmax(label))
            if image[i][j] == 255:
                filters[index][i][j] += 1

import matplotlib.pyplot as plt
normalized_filters = []
for filter in filters:
    filter /= np.amax(filter) # normalize
    filter -= filter.mean()
    filter = filter.flatten()  # become weights
    normalized_filters.append(filter)

weights = np.vstack(normalized_filters).T
print(weights.shape)
model.layers[0].set_weights([weights, np.array([0,0,0,0,0,0,0,0,0,0])])


print("checkpoint")

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

### MODEL TRAINING ###

model.fit(train_images, train_labels, epochs=10, batch_size=128, verbose=1)

### MODEL EVALUATION ###

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

### SUMMARY ###

print("Test accuracy (pretrain): {0} % ".format(round(test_acc*100, 2)))

# print()
#
# print("Weights")
# first_layer_weights = model.layers[0].get_weights()[0]
# for i, w in enumerate(first_layer_weights):
#     print("Pixel {0} weights: {1}".format(i, np.array(w)))
#
# print()
#
# print("Biases")
# first_layer_biases  = model.layers[0].get_weights()[1]
# print(first_layer_biases)
