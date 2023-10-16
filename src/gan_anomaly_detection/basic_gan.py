# https://github.com/Hourout/GAN-keras/blob/master/GAN/GAN.py

import datetime
import numpy as np
import tensorflow as tf

from src.data_pre_processing import load_data, x_width

# import tensorview as tv


def generator(latent_dim=100, image_shape=(28, 28, 1)):
    noise = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(128)(noise)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Dense(x_width, activation="tanh")(x)
    gnet = tf.keras.Model(noise, x)
    return gnet


def discriminator():
    image = tf.keras.Input(shape=(x_width,))
    # x = tf.keras.layers.Flatten()(image)
    x = tf.keras.layers.Dense(256)(image)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    logit = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    dnet = tf.keras.Model(image, logit)
    return dnet


def train(epocs=4, batch_size=128, latent_dim=100, image_shape=(28, 28, 1)):
    X_train, y_train = load_data()
    # filter out faulty boids
    X_train = X_train[np.logical_not(y_train)[:, 0], :]
    print(f"{X_train.shape = }")

    epoch_size = X_train.shape[0] // batch_size

    # image_log_interval = 50
    dnet = discriminator()
    dnet.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
        metrics=["accuracy"],
    )

    noise = tf.keras.Input(shape=(latent_dim,))
    gnet = generator(latent_dim)
    frozen = tf.keras.Model(dnet.inputs, dnet.outputs)
    frozen.trainable = False
    image = gnet(noise)
    logit = frozen(image)
    gan = tf.keras.Model(noise, logit)
    gan.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
        metrics=["accuracy"],
    )

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = "logs/" + current_time + "/train"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    with train_summary_writer.as_default():
        # tv_plot = tv.train.PlotMetrics(columns=2, wait_num=50)
        for batch in range(epoch_size * epocs):
            batch_image = X_train[
                np.random.choice(range(X_train.shape[0]), batch_size, False)
            ]
            batch_noise = np.random.normal(0, 1, (batch_size, latent_dim))
            batch_gen_image = gnet.predict(batch_noise, verbose=0)

            # if batch % image_log_interval == 0:
            #     tf.summary.image("Generator Image", batch_gen_image, step=batch)

            d_loss_real = dnet.train_on_batch(batch_image, np.ones((batch_size, 1)))
            d_loss_fake = dnet.train_on_batch(
                batch_gen_image, np.zeros((batch_size, 1))
            )
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            g_loss = gan.train_on_batch(batch_noise, np.ones((batch_size, 1)))
            tf.summary.scalar("d_loss", d_loss[0], step=batch)
            tf.summary.scalar("d_binary_acc", d_loss[1], step=batch)
            tf.summary.scalar("g_loss", g_loss[0], step=batch)
            tf.summary.scalar("g_binary_acc", g_loss[1], step=batch)
            print(
                f"Batch: {batch:>5}/{epoch_size * epocs}, D_loss: {d_loss[0]:6.4f}, D_binary_acc: {d_loss[1]:6.4f}, G_loss: {g_loss[0]:6.4f}, G_binary_acc: {g_loss[1]:6.4f}\r",
                end="",
            )
    return dnet


if __name__ == "__main__":
    dnet = train()
    dnet.save("data/dnet2.keras")
