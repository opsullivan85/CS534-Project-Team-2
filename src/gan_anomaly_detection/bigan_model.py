# Code modified from: https://github.com/jason71995/bigan/blob/master/train_mnist.py

from datetime import datetime
from src.data_pre_processing import load_data, x_width

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Concatenate, Dense, LeakyReLU, Dropout
from keras.optimizers import Adam
import numpy as np
from PIL import Image


class ModelParameters:
    # meta parameters
    latent_code_length = 100


def build_generator():
    x = Input(ModelParameters.latent_code_length)
    y = Dense(128)(x)
    y = LeakyReLU()(y)
    y = Dense(256)(y)
    y = LeakyReLU()(y)
    y = Dense(512)(y)
    y = LeakyReLU()(y)
    # y = Dropout(0.5)(y)
    y = Dense(1024)(y)
    y = LeakyReLU()(y)
    y = Dense(x_width)(y)
    return Model(x, y)


def build_encoder():
    x = Input(x_width)
    y = Dense(1024)(x)
    y = LeakyReLU()(y)
    y = Dense(512)(y)
    y = LeakyReLU()(y)
    y = Dense(256)(y)
    y = LeakyReLU()(y)
    y = Dense(128)(y)
    y = LeakyReLU()(y)
    y = Dense(ModelParameters.latent_code_length)(y)
    return Model(x, y)


def build_discriminator():
    x = Input(x_width)
    z = Input(ModelParameters.latent_code_length)
    y = Concatenate()([x, z])
    y = Dense(1024)(y)
    y = LeakyReLU()(y)
    y = Dense(512)(y)
    y = LeakyReLU()(y)
    y = Dense(256)(y)
    y = LeakyReLU()(y)
    y = Dense(128)(y)
    y = LeakyReLU()(y)
    y = Dense(1)(y)
    return Model([x, z], [y])


def build_train_step(generator, encoder, discriminator):
    g_optimizer = Adam(lr=0.0001)  # , beta_1=0.0, beta_2=0.9)
    e_optimizer = Adam(lr=0.0001)  # , beta_1=0.0, beta_2=0.9)
    d_optimizer = Adam(lr=0.0001)  # , beta_1=0.0, beta_2=0.9)

    @tf.function
    def train_step(real_image, real_code):
        tf.keras.backend.set_learning_phase(True)

        fake_image = generator(real_code)
        fake_code = encoder(real_image)

        d_inputs = [
            tf.concat([fake_image, real_image], axis=0),
            tf.concat([real_code, fake_code], axis=0),
        ]
        d_preds = discriminator(d_inputs)
        pred_g, pred_e = tf.split(d_preds, num_or_size_splits=2, axis=0)

        d_loss = tf.reduce_mean(tf.nn.softplus(pred_g)) + tf.reduce_mean(
            tf.nn.softplus(-pred_e)
        )
        g_loss = tf.reduce_mean(tf.nn.softplus(-pred_g))
        e_loss = tf.reduce_mean(tf.nn.softplus(pred_e))

        d_gradients = tf.gradients(d_loss, discriminator.trainable_variables)
        g_gradients = tf.gradients(g_loss, generator.trainable_variables)
        e_gradients = tf.gradients(e_loss, encoder.trainable_variables)

        d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
        g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
        e_optimizer.apply_gradients(zip(e_gradients, encoder.trainable_variables))

        return d_loss, g_loss, e_loss

    return train_step


def train():
    x_train, y_train = load_data()
    # filter out faulty boids
    x_train = x_train[np.logical_not(y_train)[:, 0], :]
    print(f"{x_train.shape = }")

    num_of_data = x_train.shape[0]

    check_point = 100
    batch_size = 256
    epoch_size = num_of_data // batch_size
    iters = epoch_size * 4

    z_train = np.random.uniform(
        -1.0, 1.0, (num_of_data, ModelParameters.latent_code_length)
    ).astype("float32")
    z_test = np.random.uniform(
        -1.0, 1.0, (100, ModelParameters.latent_code_length)
    ).astype("float32")

    generator = build_generator()
    encoder = build_encoder()
    discriminator = build_discriminator()
    train_step = build_train_step(generator, encoder, discriminator)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = "logs/gradient_tape/" + current_time + "/train"
    test_log_dir = "logs/gradient_tape/" + current_time + "/test"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    with train_summary_writer.as_default():
        for i in range(iters):
            real_images = x_train[np.random.permutation(num_of_data)[:batch_size]]
            real_code = z_train[np.random.permutation(num_of_data)[:batch_size]]

            d_loss, g_loss, e_loss = train_step(real_images, real_code)
            tf.summary.scalar("d_loss", d_loss, step=i)
            tf.summary.scalar("g_loss", g_loss, step=i)
            tf.summary.scalar("e_loss", e_loss, step=i)
            print(
                "\r[{}/{}]  d_loss: {:.4}, g_loss: {:.4}, e_loss: {:.4}".format(
                    i, iters, d_loss, g_loss, e_loss
                ),
                end="",
            )

            # if (i + 1) % check_point == 0:
            #     x2, y2 = load_data()
            #     y_guess = discriminator.predict([x2, encoder.predict(x2)])
            #     tf.summary.text("test", str(np.hstack([y_guess, y2])), step=i)
            #     tf.summary.text("test", str(np.hstack([y_guess, y2])), step=i)

            #     # save G(x) images
            #     image = generator.predict(encoder.predict(x_train[:100]))
            #     image = np.reshape(image, (10, 10, 28, 28))
            #     image = np.transpose(image, (0, 2, 1, 3))
            #     image = np.reshape(image, (10 * 28, 10 * 28))
            #     image = 255 * (image + 1) / 2
            #     image = np.clip(image, 0, 255)
            #     image = image.astype("uint8")
            #     Image.fromarray(image, "L").save("G_E_x-{}.png".format(i // check_point))

            #     # save G(z) images
            #     image = generator.predict(z_test)
            #     image = np.reshape(image, (10, 10, 28, 28))
            #     image = np.transpose(image, (0, 2, 1, 3))
            #     image = np.reshape(image, (10 * 28, 10 * 28))
            #     image = 255 * (image + 1) / 2
            #     image = np.clip(image, 0, 255)
            #     image = image.astype("uint8")
            #     Image.fromarray(image, "L").save("G_z-{}.png".format(i // check_point))


if __name__ == "__main__":
    train()
