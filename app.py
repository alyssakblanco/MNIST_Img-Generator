from flask import Flask, render_template
import os
import sys
from typing import Counter
import tensorflow as tf
import numpy as np
from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import random
from sklearn.utils import shuffle

app = Flask(__name__)

DB_DIR = os.path.dirname(os.path.realpath("./"))

# Import databases
sys.path.insert(1, DB_DIR)

# global variables (would eventually like to take these as inputs from the front end)
numEpochs = 1300
saveInternval = 100

# Flask webpage
@app.route('/')
def render_webpage():
    return render_template("index.html", numEpochs=numEpochs, saveInternval=saveInternval)

# improved GAN model and code from assignment 10
@app.route('/img_generator')
def img_generator():
    class GAN():
        def __init__(self, input_shape=(28,28,1), rand_vector_shape=(100,), lr=0.0002, beta=0.5):
            # input sizes
            self.image_shape = input_shape
            self.input_size = rand_vector_shape

            # optimizer
            self.opt = tf.keras.optimizers.Adam(lr,beta)

            # generator model
            self.generator = self.generator_model()
            self.generator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001, beta), metrics=['accuracy'])

             # Create Discriminator model
            self.discriminator = self.discriminator_model()
            self.discriminator.compile(loss='binary_crossentropy', optimizer = tf.keras.optimizers.Adam(0.001, beta), metrics = ['accuracy'])

            # Set the Discriminator as non trainable in the combined GAN model
            self.discriminator.trainable = False

            # Define model input and output
            input = tf.keras.Input(self.input_size)
            generated_img = self.generator(input)
            output = self.discriminator(generated_img)

            # Define and compile combined GAN model
            self.GAN = tf.keras.Model(input, output, name="GAN")
            self.GAN.compile(loss='binary_crossentropy', optimizer = self.opt, metrics=['accuracy'])

            return None

        def discriminator_model(self):
          model = tf.keras.models.Sequential(name='Discriminator')
          model.add(layers.Flatten())
          model.add(layers.Dense(units=1024, kernel_initializer='normal'))
          model.add(layers.LeakyReLU(alpha=0.02))
          model.add(layers.Dropout(0.3))
          model.add(layers.Dense(units=512, kernel_initializer='normal'))
          model.add(layers.LeakyReLU(alpha=0.02))
          model.add(layers.Dropout(0.3))
          model.add(layers.Dense(units=256, kernel_initializer='normal'))
          model.add(layers.LeakyReLU(alpha=0.02))
          model.add(layers.Dropout(0.3))
          model.add(layers.Dense(units=1, kernel_initializer='normal', activation='sigmoid'))
          return model

        def generator_model(self):
          model = tf.keras.models.Sequential(name='Generator')
          model.add(layers.Dense(units=256, kernel_initializer='normal'))
          model.add(layers.LeakyReLU(alpha=0.02))
          model.add(layers.Dense(units=512, kernel_initializer='normal'))
          model.add(layers.LeakyReLU(alpha=0.02))
          model.add(layers.Dense(units=1024, kernel_initializer='normal'))
          model.add(layers.LeakyReLU(alpha=0.02))
          model.add(layers.Dense(units=np.prod(self.image_shape), kernel_initializer='normal', activation='tanh'))
          model.add(layers.Reshape((28,28)))
          return model

        def plot_imgs(self, epoch):
            r, c = 4,4
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    noise = np.random.normal(0, 1, (1, self.input_size[0]))
                    img = self.generator.predict(noise)[0,:]
                    axs[i,j].imshow(img, cmap = plt.cm.binary)
                    axs[i,j].axis('off')
                    cnt = cnt + 1
            full_filename = 'static/imgEpoch{0}.png'
            fig.savefig(full_filename.format(epoch))
            plt.title("Epoch " + str(epoch))
            return None

        def train(self, X_train, batch_size=128, epochs=numEpochs, save_interval=100):
            print("TRAIN")
            half_batch = batch_size//2
            y_pos_train_dis = np.ones((half_batch, 1))
            y_neg_train_dis = np.zeros((half_batch, 1))
            y_train_GAN = np.ones((batch_size, 1))

            for epoch in range(epochs):
                # training data for descriminator
                X_pos_train_dis = X_train[np.random.randint(0, X_train.shape[0], half_batch)]
                X_neg_train_dis = self.generator.predict(tf.random.normal((half_batch, self.input_size[0])))

                # concat pos, negative and shuffle
                X_train_dis, y_train_dis = tf.concat([X_neg_train_dis, X_pos_train_dis], axis=0), tf.concat([y_neg_train_dis, y_pos_train_dis], axis=0)

                # trainign data for GAN model
                X_train_GAN = tf.random.normal((batch_size, self.input_size[0]))

                 # Train Discriminator
                self.discriminator.trainable = True

                loss_dis = self.discriminator.train_on_batch(X_train_dis, y_train_dis)

                self.discriminator.trainable = False
                # Train Generator
                loss_gen = self.GAN.train_on_batch(X_train_GAN, y_train_GAN)

                # Print results
                if epoch%save_interval == 0:
                    print("Discriminator loss: {0}, Generator loss: {1}".format(loss_dis[0], loss_gen[0]))
                    print("Discriminator acc.: {0}, Generator acc.: {1}".format(loss_dis[1], loss_gen[1]))
                    self.plot_imgs(epoch)

            return 0

    def choose_dataset(normalize=True):
        # load
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # normalize
        X_train = X_train / 255
        X_test = X_test / 255

        # reshape
        (X_train, y_train), (X_test, y_test) = reshape_dataset(X_train, y_train, X_test, y_test)

        return (X_train, y_train), (X_test, y_test)

    def reshape_dataset(X_train, y_train, X_test, y_test):
        """Reshape Computer Vision and Speech datasets."""

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        return (X_train, y_train), (X_test, y_test)

    gan_model = GAN()
    (X_train, _), (X_test, _) = choose_dataset()
    gan_model.train(X_train)

    return gan_model

if __name__ == '__main__':
    app.run()
