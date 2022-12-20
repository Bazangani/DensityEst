
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import cmath
from datetime import datetime
from subpixel_conv2d import SubpixelConv2D
import math

latent_dim = 49
datagen = ImageDataGenerator(rescale=1/255.0,
                             featurewise_std_normalization=True,
                             samplewise_std_normalization=True,
                             zca_whitening=True,
                             zca_epsilon=1e-06,
                             rotation_range=30,
                             width_shift_range=20.0,
                             height_shift_range=30.0,
                             shear_range=20.0,
                             zoom_range=20.0,

                             )
test_datagen = ImageDataGenerator(rescale=1 / 255.0,
                                  featurewise_std_normalization=True,
                                  samplewise_std_normalization=True,
                                  zca_whitening=True,
                                  zca_epsilon=1e-06,
                                  rotation_range=30,
                                  width_shift_range=20.0,
                                  height_shift_range=30.0,
                                  shear_range=20.0,
                                  zoom_range=20.0,
                                  )

# importing non label data
train_ds = tf.keras.preprocessing.image_dataset_from_directory('C:/Users/farideh/Desktop/Rnadom field/VAE and RF/DATA/',
                                                       image_size=(100, 100),
                                                       color_mode='grayscale',
                                                       shuffle=True,
                                                       batch_size=100,
                                                       label_mode=None)

val_ds = tf.keras.preprocessing.image_dataset_from_directory('C:/Users/farideh/Desktop/Rnadom field/VAE and RF/DATA/',
                                                     image_size=(100, 100),
                                                     color_mode='grayscale',
                                                     shuffle=True,
                                                     batch_size=1,
                                                     label_mode=None)

def process(image):
    image = tf.cast(image/255., tf.float32)
    return image

AUTOTUNE = tf.data.AUTOTUNE
# keep the images in memory (performant on-disk cache)
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)




# for images in train_ds:
#     for i in range(4):
#         ax =plt.subplot(2,2,i+1)
#
#         plt.imshow(images[i])
#         plt.axis("off")
# plt.show()

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        C = inputs
        # we have to adapt the size
        batch_size = tf.shape(inputs)[0]
        lat_dim = tf.shape(inputs)[1]
        #C = tf.reshape(C, [ -1, 7, 7])
        #L = tf.linalg.cholesky(C)
        epsilon = tf.keras.backend.random_normal(shape=(batch_size, lat_dim,lat_dim))
        Y = C * epsilon
        Y = tf.convert_to_tensor(Y, dtype='float32')
        return Y, C

# convolution block
def Conv_block(n_channel):
    conv_block = tf.keras.Sequential(
        [
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(n_channel, 3, activation="relu", strides=1, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(n_channel, 3, activation="relu", strides=1, padding="same"),
            ]
    )
    return conv_block

encoder_inputs = keras.Input(shape=(100, 100, 1))
#encoder_inputs = tf.keras.layers.Normalization(mean=0.0, variance=1.0)(encoder_inputs)

# encoder
x1 = Conv_block(1)(encoder_inputs)
x1_res = x1 + encoder_inputs
x1_res = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(x1_res)  # 50 x 50
x1_res_chanel = tf.keras.layers.Conv2D(16, 3, activation="relu", strides=1, padding="same")(x1_res)  # increase the channel to 16


x2 = Conv_block(16)(x1_res_chanel)
x2_res = x1_res_chanel + x2
x2_res = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(x2_res)  # 25 x 25
x2_res_chanel = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=1, padding="same")(x2_res)  # increase the channel to 32


x3 = Conv_block(32)(x2_res_chanel)
x3_res = x3 + x2_res_chanel
x3_res = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(x3_res)  # 13 x 13
x3_res_chanel = tf.keras.layers.Conv2D(64, 3, activation="relu", strides=1, padding="same")(x3_res)  # increase the channel to 64


x4 = Conv_block(64)(x3_res_chanel)
x4_res = x4 + x3_res_chanel
x4_res = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(x4_res)  # 7 x 7
x4_res_chanel = tf.keras.layers.Conv2D(128, 3, activation="relu", strides=1, padding="same")(x4_res)

z_log_var = layers.Conv2D(1, 3, activation="relu", strides=1, padding="same", name="z_log_var")(x4_res_chanel)

# normal distribution and sampling

batch_covariance, z = Sampling()([z_log_var])
encoder = keras.Model(encoder_inputs, [batch_covariance, z], name="encoder")
encoder.summary()


# Decoder
latent_inputs = keras.Input(shape=(7, 7,))
x = layers.Dense(7 * 7 * 128, activation="relu")(latent_inputs)
x1 = layers.Reshape((7, 7, 896))(x)

# conv Transpose
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x1)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)  # 28 x 28
x = layers.Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(x)  # 56 x 56
x = y = SubpixelConv2D(upsampling_factor=2)(x)  # 112 x 112

#
# # pixel shuffle
# y = SubpixelConv2D(upsampling_factor=2)(x1)  # 14x14
# y = layers.Conv2D(64, 1, activation="relu", strides=1, padding="same")(y)  # @64 14x14
# y = SubpixelConv2D(upsampling_factor=2)(y)  # 28x28
# y = layers.Conv2D(32, 1, activation="relu", strides=1, padding="same")(y)  # @32 28x28
# y = SubpixelConv2D(upsampling_factor=2)(y)  # 56x56
# y = layers.Conv2D(16, 1, activation="relu", strides=1, padding="same")(y)  # @16 56x56
# y = SubpixelConv2D(upsampling_factor=2)(y)  # 112x112
# y = layers.Conv2D(1, 1, activation="relu", strides=1, padding="same")(y)  # @1 112x112

mix_out = x

decoder_outputs = layers.Conv2D(1, 7, activation="relu", strides=1, padding="valid")(mix_out)   # 106 x 106
decoder_outputs = layers.Conv2D(1, 7, strides=1, activation="softmax", padding="valid")(decoder_outputs)   # 100 x 100

decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

decoder.summary()

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:


            z, z_log_var = self.encoder(data)


            reconstruction = self.decoder(z)

            tf.print(reconstruction, [reconstruction])
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.mean_squared_logarithmic_error(data, reconstruction), axis=(1, 2))
            )
            kl_loss = -0.5 * ( z_log_var - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            kl_loss = tf.cast(kl_loss, dtype='float32')
            total_loss = reconstruction_loss + kl_loss
            tf.print(total_loss,[total_loss])

        grads = tape.gradient(total_loss, self.trainable_weights)
        grads = tf.distribute.get_replica_context().all_reduce('sum', grads)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights),
                                       experimental_aggregate_gradients=False)

        self.total_loss_tracker.update_state(total_loss)

        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
vae = VAE(encoder, decoder)

vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

checkpoint_path = "model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create a callback that saves the model's weights

#vae.load_weights(checkpoint_path)
vae.fit(train_ds, epochs=10000, batch_size=2000, shuffle=True)

#plot_model(decoder, to_file='model_plot.png', show_shapes=True,show_layer_names=True)
fig, axs = plt.subplots(nrows=2, ncols=6)

# apply the model

for i, image in tqdm(enumerate(val_ds)):

    z , z_log_var = vae.encoder.predict(image)  # get the parameter of the hidden space
    decoded_info = vae.decoder.predict(z)
    img = (np.array(image[0, :, :, 0])).reshape(100, 100)
    recons_img = decoded_info[0].reshape(100, 100)
    axs[0, i].imshow(img)
    axs[0, i].set_title("real")
    axs[1, i].imshow(recons_img)
    axs[1, i].set_title("reconstructed")
    if i == 5:
        break
plt.show()


def plot_latent_space(vae, n=5, figsize=30):
    # display a n*n 2D manifold of digits
    digit_size = 100
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    print(grid_y)
    print(grid_x)

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="hot")
    plt.show()


plot_latent_space(vae)


