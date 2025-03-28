import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plot

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.90):
          print("\nReached 90% accuracy so cancelling training!")
          self.model.stop_training = True
#want to get rid of the law of diminishing returns
callbacks = myCallback()
fashion_mnist = keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()
#trainning images are all images in the dataset basically our input
#training labels are the classes of clothing.
#Numbers are used for labels since computers understand them better and we avoid bias towards a specific language
#test images are the images in the dataset that are shoes basically the output
plot.imshow(training_images[57800])
plot.show()
print(training_labels[57800])
print(training_images[57800])
#print(test_labels[57800])

#normalizing
training_images = training_images / 255.0
test_images = test_images / 255.0
model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),keras.layers.Dense(1024, activation=tf.nn.relu),keras.layers.Dense(10, activation=tf.nn.softmax)])
#Sequential = That defines a sequence of layers in the neural network
#Flatter = turns the image into a 1D set
#Dense = adds a layer of neurons

#input shape tells us the shape of the input in this case its 28px X 28px
#10 represents the different classes of output.
#model takes in a variety of images and outputs a number from 1 - 10
#128 different functions. when the image is loaded, it needs to determine the value

model.compile(optimizer = tf.keras.optimizers.Adam(),loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
#neural network is initiallized with random values
#loss function will measure how good or bad the results were
#with optimizer it will generate new parameters for the functions to see if it can lower that number

#activations functions. first one is called relu or rectified linear unit.
#if (x > 0){ return x;} else{return 0;} only passes values of 0 or greater to the next unit.
                          
#second one is softmax which picks the biggest number in the set. in our case this is probability. so when the functions return something its going to be an array of different probabilities and item can be but softmax makes it into 1 number.

model.fit(training_images, training_labels, epochs = 100, callbacks=[callbacks])

model.evaluate(test_images, test_labels)
#returns percent accuracy
#these are images our model has previously see so we can use them to see how well our model performs.

classifications = model.predict(test_images)

print(classifications[7800])
print(test_labels[7800])
plot.imshow(test_images[7800])
plot.show()

#these are images our model has not previously see so we can use them to see how well our model performs.

