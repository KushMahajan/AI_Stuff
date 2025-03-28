import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorflow import keras
import numpy as np

model = keras.Sequential([keras.layers.Dense(units=1, input_shape = [1])])
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype = float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype = float)

model.fit(xs,ys, epochs = 1000)

print(model.predict(np.array([10.0])))


#model is a neural network. this is the simplest neural network which has 1 nueron.
#loss function tells how bad that guess is, and then optimizer function tries to minimize the loss
#mean squared error is the sum of (Yactual - Ypredicted)^2/number of terms
