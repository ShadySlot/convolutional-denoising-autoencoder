# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

## Convolution Autoencoder Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:  Setup the data


### STEP 2: Prepare the data

### STEP 3: Build the autoencoder,use the Functional API to build our convolutional autoencoder.

### STEP 4 : Now we can train our autoencoder using train_data as both our input data and target. Notice we are setting up the validation data using the same format.

### STEP 5 : Let's predict on our test dataset and display the original image together with the prediction from our autoencoder.

Notice how the predictions are pretty close to the original images, although not quite the same.

### STEP 6 : Now that we know that our autoencoder works, let's retrain it using the noisy data as our input and the clean data as our target. We want our autoencoder to learn how to denoise the images.
Write your own steps

## PROGRAM

```sh
developed by: r nitish kumar
Register no.: 212219220036
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
(x_train, _), (x_test, _) = mnist.load_data()
x_train.shape
x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))
noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_te
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
ax = plt.subplot(1, n, i)
plt.imshow(x_test_noisy[i].reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
input_img = keras.Input(shape=(28, 28, 1))
# Write your encoder here
x=layers.Conv2D(32,(3,3),activation='relu',padding='same')(input_img)
x=layers.MaxPooling2D((2, 2), padding='same')(x)
x=layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
# Encoder output dimension is ## Mention the dimention ##
# Write your decoder here
x=layers.Conv2D(32,(3,3),activation='relu',padding='same')(encoded)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = keras.Model(input_img, decoded)
autoencoder.summary()autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train_scaled,
epochs=2,
batch_size=128,
shuffle=True,
validation_data=(x_test_noisy, x_test_scaled))
import pandas as pd
metrics = pd.DataFrame(autoencoder.history.history)
metrics.head()
metrics[['loss','val_loss']].plot()
decoded_imgs = autoencoder.predict(x_test_noisy)
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
# Display original
ax = plt.subplot(3, n, i)
plt.imshow(x_test_scaled[i].reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
# Display noisy
ax = plt.subplot(3, n, i+n)
plt.imshow(x_test_noisy[i].reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
# Display reconstruction
ax = plt.subplot(3, n, i + 2*n)
plt.imshow(decoded_imgs[i].reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
Normalize and reshape the data
train_data = preprocess(train_data) test_data = preprocess(test_data)
Create a copy of the data with added
noise
noisy_train_data = noise(train_data) noisy_test_data = noise(test_data)
Display the train data and a version of it
with added noise
display(train_data, noisy_train_data) input = layers.Input(shape=(28, 28, 1))

Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input) x =
layers.MaxPooling2D((2, 2), padding="same")(x) x = layers.Conv2D(32, (3, 3),
activation="relu", padding="same")(x) x = layers.MaxPooling2D((2, 2),
padding="same")(x)

Decoder
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.subplot(2, n, i + 1 + n)
plt.imshow(image2.reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x) x
= layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x) x
= layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

Autoencoder
autoencoder = Model(input, x) autoencoder.compile(optimizer="adam",
loss="binary_crossentropy") autoencoder.summary() autoencoder.fit( x=train_data,
y=train_data, epochs=50, batch_size=128, shuffle=True, validation_data=(test_data,
test_data), ) predictions = autoencoder.predict(test_data) display(test_data, predictions)
autoencoder.fit( x=noisy_train_data, y=train_data, epochs=100, batch_size=128,
shuffle=True, validation_data=(noisy_test_data, test_data), ) predictions =
autoencoder.predict(noisy_test_data) display(noisy_test_data, predictions)
```

## OUTPUT
![image](https://user-images.githubusercontent.com/112503943/203128183-e9ba46cb-43dc-469f-9e9d-990c1b8c1272.png)


![image](https://user-images.githubusercontent.com/112503943/203128070-6d906882-9efa-4417-a6da-f15e2d651066.png)


![image](https://user-images.githubusercontent.com/112503943/203128272-bb843d72-380f-4df0-af30-519938a04a04.png)



## RESULT

Thus the convolutional-denoising-autoencoder has been implemented successfully and the output is verified.
