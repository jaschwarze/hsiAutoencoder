import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / np.max(x_train)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_train = x_train.astype("float32")

training_dataset = tf.data.Dataset.from_tensor_slices(x_train)
training_dataset = training_dataset.batch(batch_size=32)
training_dataset = training_dataset.shuffle(x_train.shape[0])


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_neurons):
        super(Encoder, self).__init__()

        self.hidden_layer1 = tf.keras.layers.Dense(
            units=num_neurons,
            activation=tf.nn.relu)

        self.hidden_layer2 = tf.keras.layers.Dense(
            units=num_neurons,
            activation=tf.nn.relu)

    def call(self, input_features):
        """Call function for the decoder class

        @param input_features: input features
        @return output: reconstructed input features
        """
        activation = self.hidden_layer1(input_features)
        return self.hidden_layer2(activation)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_neurons, mnist_dim):
        super(Decoder, self).__init__()

        self.hidden_layer = tf.keras.layers.Dense(
            units=num_neurons,
            activation=tf.nn.relu)

        self.output_layer = tf.keras.layers.Dense(
            units=mnist_dim,
            activation=tf.nn.relu)

    def call(self, input_data):
        """Call function for the decoder class

        @param input_data: input features
        @return output: reconstructed input features
        """
        activation = self.hidden_layer(input_data)
        return self.output_layer(activation)


class Autoencoder(tf.keras.Model):
    def __init__(self, num_neurons, mnist_dim):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(num_neurons)

        self.decoder = Decoder(num_neurons, mnist_dim)

        # Define an optimizer
        self.opt = tf.optimizers.Adam(learning_rate=0.01)

    def call(self, x_train):
        """Call function for the autoencoder class

        @param x_train: input features
        @return output: reconstructed input features
        """

        latent_representation = self.encoder(x_train)
        output = self.decoder(latent_representation)

        return output

    def compute_loss(self, x_train):
        """Compute MSE loss function.

        @param x_train: input features
        """

        mse = tf.keras.losses.MeanSquaredError()
        loss = mse(x_train, self(x_train))
        return loss

    def train(self, x_train):
        """Train the autoencoder.

        @parameter x_train: training input features
        """

        # Compute the gradients and apply the gradient descent step
        with tf.GradientTape() as tape:
            gradients = tape.gradient(self.compute_loss(x_train), self.trainable_variables)
            gradient_variables = zip(gradients, self.trainable_variables)
            self.opt.apply_gradients(gradient_variables)


autoencoder = Autoencoder(num_neurons=64, mnist_dim=784)
num_epoch = 5

for epoch in range(num_epoch):

    # Iterate over the batches
    for step, x_train in enumerate(training_dataset):
        autoencoder.train(x_train)
        loss_values = autoencoder.compute_loss(x_train)
        print(loss_values)
