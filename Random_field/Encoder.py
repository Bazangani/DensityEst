import tensorflow as tf
from tensorflow import keras
from RandomField import Sampling


def Conv_block(n_channel):
    conv_block = tf.keras.Sequential(
        [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(n_channel, 3, activation="relu", strides=1, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(n_channel, 3, activation="relu", strides=1, padding="same"),

        ]
    )


class Encoder(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.MAX_pool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')
        self.x1_res_chanel = tf.keras.layers.Conv2D(16, 3, activation="relu", strides=1, padding="same")
        self.x2_res_chanel = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=1, padding="same")
        self.x3_res_chanel = tf.keras.layers.Conv2D(64, 3, activation="relu", strides=1, padding="same")
        self.x4_res_chanel = tf.keras.layers.Conv2D(128, 3, activation="relu", strides=1, padding="same")
        self.Flatten = tf.keras.layers.Flatten()
        self.Dense = tf.keras.layers.Dense(7 * 7 * 128, activation="relu")

    def call(self, inputs):
            inputs = self.inputs
            x1 = Conv_block(1)(inputs)
            x1_res = x1 + inputs
            x1_res = self.MAX_pool(x1_res)  # 50 x 50
            x1_res_chanel = self.x1_res_chanel(x1_res)  # increase the channel to 16

            x2 = Conv_block(16)(x1_res_chanel)
            x2_res = x1_res_chanel + x2
            x2_res = self.MAX_pool(x2_res)  # 25 x 25
            x2_res_chanel = self.x2_res_chanel(x2_res)  # increase the channel to 32

            x3 = Conv_block(32)(x2_res_chanel)
            x3_res = x3 + x2_res_chanel
            x3_res = self.MAX_pool(x3_res)  # 13 x 13
            x3_res_chanel = self.x3_res_chanel(x3_res)  # increase the channel to 64

            x4 = Conv_block(64)(x3_res_chanel)
            x4_res = x4 + x3_res_chanel
            x4_res = self.MAX_pool(x4_res)  # 7 x 7
            x4_res_chanel = self.x4_res_chanel(x4_res)

            x = self.Flatten(x4_res_chanel)
            x = self.Dense(x)

            # normal random field with zero mean and eigenvector
            eigenvalues = tf.keras.layers.Dense(49, name="eigen_values")(x)  # batch_size x latent_dim

            batch_covariance, Y = Sampling()([eigenvalues])
            return batch_covariance, Y
