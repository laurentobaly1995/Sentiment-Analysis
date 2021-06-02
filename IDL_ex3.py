from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras import activations
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import losses
import random
import numpy as np
import os

latent_dim = 64
noise_sigma = 0.35
train_AE = True
sml_train_size = 50
BUFFER_SIZE = 60000
BATCH_SIZE = 64
NOISE_DIM = 64
NUM_DIGITS = 10

# load train and test images, and pad & reshape them to (-1,32,32,1)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)).astype('float32') / 255.0
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1)).astype('float32') / 255.0
x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)))
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)))
print(x_train.shape)
print(x_test.shape)
exit()
y_train = keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test = keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')

encoder = Sequential()
encoder.add(layers.Conv2D(16, (4, 4), strides=(2, 2), activation='relu', padding='same', input_shape=(32, 32, 1)))
encoder.add(layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same'))
encoder.add(layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same'))
encoder.add(layers.Conv2D(96, (3, 3), strides=(2, 2), activation='relu', padding='same'))
encoder.add(layers.Reshape((2 * 2 * 96,)))
encoder.add(layers.Dense(latent_dim))

# at this point the representation is (4, 4, 8) i.e. 128-dimensional
decoder = Sequential()
decoder.add(layers.Dense(2 * 2 * 96, activation='relu', input_shape=(latent_dim,)))
decoder.add(layers.Reshape((2, 2, 96)))
decoder.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same'))
decoder.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same'))
decoder.add(layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), activation='relu', padding='same'))
decoder.add(layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), activation='sigmoid', padding='same'))

autoencoder = keras.Model(encoder.inputs, decoder(encoder.outputs))
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

checkpoint_path = "model_save/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True)

if train_AE:
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True)
    autoencoder.fit(x_train + noise_sigma * np.random.randn(*x_train.shape), x_train,
                    epochs=1,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[cp_callback])
else:
    autoencoder.load_weights(checkpoint_path)

decoded_imgs = autoencoder.predict(x_test)
latent_codes = encoder.predict(x_test)
decoded_imgs = decoder.predict(latent_codes)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# Classifer Network
classifier = Sequential()
classifier.add(layers.Dense(32, activation='relu', input_shape=(latent_dim,)))
classifier.add(layers.Dense(10, activation='softmax'))

train_codes = encoder.predict(x_train[:sml_train_size])
test_codes = encoder.predict(x_test)

classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

classifier.fit(train_codes, y_train[:sml_train_size],
               epochs=2,
               batch_size=16,
               shuffle=True,
               validation_data=(test_codes, y_test), callbacks=[cp_callback])

full_cls_enc = keras.models.clone_model(encoder)
full_cls_cls = keras.models.clone_model(classifier)
full_cls = keras.Model(full_cls_enc.inputs, full_cls_cls(full_cls_enc.outputs))

full_cls.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

full_cls.fit(x_train[:sml_train_size], y_train[:sml_train_size],
             epochs=1,
             batch_size=16,
             shuffle=True,
             validation_data=(x_test, y_test))

# defining the means in which we'll evaluate the model
criterion = keras.losses.BinaryCrossentropy(from_logits=True)
accuracy_calc = tf.keras.metrics.BinaryAccuracy()
optimizer = keras.optimizers.Adam()


def create_batches(x, y):
    """
    Returns a batches of shuffeled data of BATCH_SIZE size.
    """
    return tf.data.Dataset.from_tensor_slices((x, y)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def gen():
    """
    Returns a generator built from a MLP architecture network.
    """

    generator = Sequential()
    generator.add(layers.Dense(NOISE_DIM, activation='relu', input_shape=(NOISE_DIM,)))
    generator.add(layers.BatchNormalization())
    generator.add(layers.Dense(256, activation='relu'))
    generator.add(layers.BatchNormalization())
    generator.add(layers.Dense(100, activation='relu'))
    generator.add(layers.BatchNormalization())
    generator.add(layers.Dense(latent_dim))

    return generator


def conditional_gen():
    """
    Returns a generator built from a MLP architecture network for the conditional case.
    """
    generator = Sequential()
    generator.add(layers.Dense(NOISE_DIM + NUM_DIGITS, activation='relu', input_shape=(NOISE_DIM + NUM_DIGITS,)))
    generator.add(layers.BatchNormalization())
    generator.add(layers.Dense(256, activation='relu'))
    generator.add(layers.BatchNormalization())
    generator.add(layers.Dense(100, activation='relu'))
    generator.add(layers.BatchNormalization())
    generator.add(layers.Dense(latent_dim))

    return generator


def disc():
    """
    Returns a discriminator built from a MLP architecture network.
    """
    discriminator = Sequential()
    discriminator.add(layers.Dense(128, activation='relu', input_shape=(latent_dim,)))
    discriminator.add(layers.Dense(1))

    return discriminator


def conditional_disc():
    """
    Returns a discriminator built from a MLP architecture network for the conditional case.
    """
    discriminator = Sequential()
    discriminator.add(layers.Dense(128, activation='relu', input_shape=(latent_dim + NUM_DIGITS,)))
    discriminator.add(layers.Dense(1))

    return discriminator


def disc_loss(true_images, false_images):
    """
    Calculates the binary cross antropy. On true images the discriminator should
    predict 1, and on fake images (false images) it should predict 0.
    """
    true_images_loss = criterion(tf.ones_like(true_images), true_images)
    false_images_loss = criterion(tf.zeros_like(false_images), false_images)

    return true_images_loss + false_images_loss


def gen_loss(false_images):
    """
    Calculates the binary cross antropy. On fake images (false images) it should
    predict 1 to trick the discriminator.
    """
    false_images_loss = criterion(tf.ones_like(false_images), false_images)

    return false_images_loss


def train(training_data):
    """
    Train the generator and discrimonator according the GAN's prosidure. Returns
    the generator.
    """

    generator = gen()
    generator_optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.4)
    generator_loss = tf.keras.metrics.BinaryCrossentropy(from_logits=True)
    generator_acc = tf.keras.metrics.BinaryAccuracy()

    discriminator = disc()
    discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.4)
    discriminator_loss_true_img = tf.keras.metrics.BinaryCrossentropy(from_logits=True)
    discriminator_loss_false_img = tf.keras.metrics.BinaryCrossentropy(from_logits=True)
    discriminator_acc = tf.keras.metrics.BinaryAccuracy()

    for epoch in range(1):

        for batch, labels in training_data:
            noise_matrix = tf.random.normal([BATCH_SIZE, NOISE_DIM])
            latent_encoder_vec = encoder.predict(batch)

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images_vec = generator(noise_matrix, training=True)

                real_output = discriminator(latent_encoder_vec, training=True)
                fake_output = discriminator(generated_images_vec, training=True)

                loss_gen = gen_loss(fake_output)
                loss_disc = disc_loss(real_output, fake_output)

                # calculating loss:
                generator_loss.update_state(tf.ones_like(fake_output), tf.sigmoid(fake_output))
                discriminator_loss_false_img.update_state(tf.zeros_like(fake_output), tf.sigmoid(fake_output))
                discriminator_loss_true_img.update_state(tf.ones_like(real_output), tf.sigmoid(real_output))

                # calculating accuracy:
                true_labels = tf.concat([tf.ones_like(real_output), tf.zeros_like(fake_output)], 0)
                discriminator_labels = tf.concat([real_output, fake_output], 0)
                discriminator_acc.update_state(true_labels, tf.sigmoid(discriminator_labels))
                generator_acc.update_state(tf.ones_like(fake_output), tf.sigmoid(fake_output))

            gradients_of_generator = gen_tape.gradient(loss_gen, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(loss_disc, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        # plot imgaes using the decoder:
        fake_images = decoder.predict(generated_images_vec)
        num = 10
        plt.figure(figsize=(20, 4))
        for i in range(1, num + 1):
            ax = plt.subplot(2, n, i)
            plt.imshow(fake_images[i].reshape(32, 32))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()

        print("discriminator accuracy: ", discriminator_acc.result())
        print("generator accuracy: ", generator_acc.result())

        print("discriminator loss on fake images: ", discriminator_loss_false_img.result())
        print("discriminator loss on real images: ", discriminator_loss_true_img.result())
        print("generator loss: ", generator_loss.result())

        generator_loss.reset_states()
        generator_acc.reset_states()
        discriminator_loss_false_img.reset_states()
        discriminator_loss_true_img.reset_states()
        discriminator_acc.reset_states()

    return generator


def interpolating(l1, l2, title):
    """
    Given two latent vectors and a title, calculates the convex linear combination
    and plots 10 images on the connecting line using the decoder with the
    relavent title.
    """

    steps = np.linspace(0, 1, 11)
    images = []
    for i in steps:
        vec = i * l2 + (1 - i) * l1
        images.append(vec)

    decoded_images = decoder.predict(np.array(images))

    num = 10
    plt.figure(figsize=(20, 4))
    for i in range(0, num):
        # Display original
        ax = plt.subplot(1, num, i + 1)
        plt.imshow(decoded_images[i].reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.title(title, fontsize=15)

    plt.show()


def train_conditional_gan(training_data):
    """
    Train the generator and discrimonator according the GAN's prosidure.
    This model learns to distinguish different digits, using one hot vectors being
    attached to the latent vectors/noise which represent a digit.
    Returns the generator.
    """

    generator = conditional_gen()
    generator_optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.4)
    generator_loss = tf.keras.metrics.BinaryCrossentropy(from_logits=True)
    generator_acc = tf.keras.metrics.BinaryAccuracy()

    discriminator = conditional_disc()
    discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.4)
    discriminator_acc = tf.keras.metrics.BinaryAccuracy()

    for epoch in range(1):

        for batch, labels in training_data:
            noise_matrix = tf.random.normal([BATCH_SIZE, NOISE_DIM])
            latent_encoder_vec = encoder.predict(batch)

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                random_digits = np.random.randint(0, 9, BATCH_SIZE)
                fake_one_hot_vectors = tf.one_hot(random_digits, NUM_DIGITS)
                real_one_hot_vectors = labels
                noise_with_digits = tf.concat([noise_matrix, fake_one_hot_vectors], axis=1)
                generated_images_vec = generator(noise_with_digits, training=True)

                real_output = discriminator(tf.concat([latent_encoder_vec, real_one_hot_vectors], axis=1),
                                            training=True)
                fake_output = discriminator(tf.concat([generated_images_vec, fake_one_hot_vectors], axis=1),
                                            training=True)

                loss_gen = gen_loss(fake_output)
                loss_disc = disc_loss(real_output, fake_output)

                # calculating loss:
                generator_loss.update_state(tf.ones_like(fake_output), tf.sigmoid(fake_output))

                # calculating accuracy:
                true_labels = tf.concat([tf.ones_like(real_output), tf.zeros_like(fake_output)], 0)
                discriminator_labels = tf.concat([real_output, fake_output], 0)
                discriminator_acc.update_state(true_labels, tf.sigmoid(discriminator_labels))
                generator_acc.update_state(tf.ones_like(fake_output), tf.sigmoid(fake_output))

            gradients_of_generator = gen_tape.gradient(loss_gen, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(loss_disc, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        # plot all digits at increasing order using the decoder:

        noise = tf.random.normal([50, NOISE_DIM])
        fake_one_hot_vectors = tf.one_hot(list(range(0, 10)) * 5, 10)
        generated_latent_vector = generator(tf.concat([noise, fake_one_hot_vectors], axis=1), training=True)
        fake_images = decoder.predict(generated_latent_vector)

        num = 50
        plt.figure(figsize=(20, 4))
        for i in range(1, num + 1):
            # Display original
            ax = plt.subplot(5, 10, i)
            plt.imshow(fake_images[i - 1].reshape(32, 32))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()

        print("discriminator accuracy: ", discriminator_acc.result())
        print("generator accuracy: ", generator_acc.result())
        print("generator loss: ", generator_loss.result())

        generator_loss.reset_states()
        generator_acc.reset_states()
        discriminator_acc.reset_states()

    return generator


# Question 3:

training_data = create_batches(x_train, y_train)
gen_q3 = train(training_data)

# Interpolating :

noise1 = tf.random.normal([1, NOISE_DIM])
noise2 = tf.random.normal([1, NOISE_DIM])
l1 = gen_q3.predict(noise1)
l2 = gen_q3.predict(noise2)

interpolating(l1, l2, "Interpolating from GAN's latent space")

# Note: we could have juct taken x_test[0], x_test[1] since they differ, however
# we wanted to see many diffent options and therefor exicuted the following code
# many times (risking the fact that the two digits will be the same, but with
# high probablity we get at least one "good" intepolation)

first_img = x_test[random.randint(0, 10000)].reshape(1, 32, 32, 1)
second_img = x_test[random.randint(0, 10000)].reshape(1, 32, 32, 1)
l1 = encoder.predict(first_img)
l2 = encoder.predict(second_img)

interpolating(l1, l2, "Interpolating from AE's latent space")

# Question 4:

gen_q4 = train_conditional_gan(training_data)

