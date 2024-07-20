
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from keras.src.datasets import mnist            # Mnist library
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normilize input data
x_train = x_train / 255
x_test = x_test / 255

# ( Example: y_train = 5, so we need to change data, after that we have massive [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] )
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Show first 25 images from test data
# plt.figure(figsize=(10,5))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_train[i], cmap=plt.cm.binary)
#plt.show()


# Build NN's structure
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(400, activation='relu'),
    Dense(10, activation='softmax')                 # Output result function
])

#print(model.summary())      # Print of NN's structure to console


model.compile(optimizer='adam',
              loss='categorical_crossentropy',      # Because we choose 'softmax' in output result function
              metrics=['accuracy'])

# Start training process
model.fit(x_train,y_train_cat,
          batch_size=16,                # after all 32 images we start to correct weight odds
          epochs=5,
          validation_split=0.2)         # 80% of train data   and   20% of validation data (like a test data but in training)


# Testing our trained model
print("\nTesting model accuracy:")
model.evaluate(x_test, y_test_cat)

# Test on choosen image
print("\nTest on choosen image")
n = 1
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print("\nResult: ", res)
print(f"number: {np.argmax(res)}")

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

model.save("my_model.keras")
