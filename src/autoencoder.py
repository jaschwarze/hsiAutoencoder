import keras
from keras.models import Model
from keras.layers import Input, Dense
import tensorflow as tf
from src.config import *
import numpy as np

def frobenius_loss(y_true, y_pred):
    m = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
    frobenius_norm = tf.norm(y_true - y_pred, ord="fro", axis=[-2, -1])
    return frobenius_norm / (2 * m)


def row_sparse_regularization(matrix):
    return tf.reduce_sum(tf.norm(matrix, ord=2, axis=1))


class Autoencoder(Model):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

        input_layer = Input(shape=(self.input_dim,), name="input_layer")
        hidden_layer = Dense(self.encoding_dim, activation="sigmoid", name="hidden_layer")(input_layer)
        output_layer = Dense(self.input_dim, activation="linear", name="output_layer")(hidden_layer)

        self.autoencoder = Model(input_layer, output_layer)
        self.encoder = Model(input_layer, hidden_layer)

        encoded_input = keras.Input(shape=(encoding_dim,), name="encoded_input")
        decoder_layer = self.autoencoder.layers[-1]
        self.decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
        self.autoencoder.compile(optimizer="adam", loss=self.combined_loss)

    def combined_loss(self, y_true, y_pred):
        return (
                frobenius_loss(y_true, y_pred)
                + ALPHA * row_sparse_regularization(self.autoencoder.layers[1].weights[0])
                + (BETA / 2) * (tf.norm(self.autoencoder.layers[1].weights[0], ord="fro", axis=[-2, -1]) +
                                tf.norm(self.autoencoder.layers[2].weights[0], ord="fro", axis=[-2, -1]))
        )

    def train(self, data, epochs, batch_size, shuffle, validation_data=None):
        return self.autoencoder.fit(x=data, y=data, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
                                    validation_data=validation_data)

    def call(self, inputs, training=None, mask=None):
        encoded = self.encoder(inputs)
        return self.decoder(encoded)

    def encode(self, image):
        return self.encoder.predict(image)

    def decode(self, encoded_image):
        return self.decoder.predict(encoded_image)

    def save_encoder(self, filepath, overwrite=True, save_format=None, **kwargs):
        self.encoder.save(filepath=filepath, overwrite=overwrite, save_format=save_format)
