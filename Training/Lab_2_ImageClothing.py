import tensorflow as tf

# ------------ Callback to stop epochs on a condition
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.90):
            print("\n  Reached 90% accuracy so. Cancelling training.")
            self.model.stop_training = True

callbacks = myCallback()

# ------------ DATA
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# ------------- Print DATA
#import matplotlib.pyplot as plt
#plt.imshow(training_images[0])
#print(training_labels[0])
#print(training_images[0])

# ------------- Normalize Data (doesn't really change accurancy or speed in this test case)
training_images = training_images / 255.0
test_images = test_images / 225.0

# -------------- MODEL
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# -------------- Compile and TRAIN
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])

# -------------- TEST
model.evaluate(test_images, test_labels)
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])