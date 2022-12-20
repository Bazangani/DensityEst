
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime
from VAE import VAE
from Decoder import Decoder
from Encoder import Encoder
import numpy as np

tf.debugging.enable_check_numerics()

latent_dim = 12
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



#train_ds = tf.keras.preprocessing.image_dataset_from_directory('C:/Users/farideh/Desktop/Rnadom field/Random_Var/Dataset/TF_loader/train',
#                                                       image_size=(100, 100),
#                                                       color_mode='grayscale',
#                                                       shuffle=True,
#                                                       batch_size=1000,
#                                                       label_mode=None)


train_ds = tf.keras.preprocessing.image_dataset_from_directory('C:/Users/farideh/Desktop/Rnadom field/VAE and RF/DATA/',
                                                       image_size=(100, 100),
                                                       color_mode='grayscale',
                                                       shuffle=True,
                                                       batch_size=1000,
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


logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

vae = VAE(Encoder, Decoder)





vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
checkpoint_path = "model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create a callback that saves the model's weights
vae.fit(train_ds, epochs=10000, batch_size=2000, shuffle=True)
tf.debugging.disable_check_numerics()

#plot_model(decoder, to_file='model_plot.png', show_shapes=True,show_layer_names=True)
fig, axs = plt.subplots(nrows=2, ncols=6)
# apply the model

for i, image in tqdm(enumerate(val_ds)):

    z_mean, z_log_var, z = vae.encoder.predict(image)  # get the parameter of the hidden space
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


# the output of the decoder is nan value therefore the loss is nan as well and the weights are probably bad too
# check the initioalization of the weights and anlso th outpu of the network