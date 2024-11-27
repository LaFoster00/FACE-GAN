from keras import models, random, metrics, losses, layers, ops
import tensorflow as tf
import numpy as np
from wandb.docker import image_id


class FaceGAN(models.Model):
    def __init__(self, discriminator, generator, latent_dim, image_dim, num_classes):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.image_dim = image_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.seed_generator = random.SeedGenerator(1337)
        self.gen_loss_tracker = metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn, run_eagerly=False):
        super().compile(run_eagerly=run_eagerly)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data.
        real_images, labels = data

        age_labels = labels['age_output']
        gender_labels = labels['gender_output']
        labels = ops.stack([age_labels, gender_labels], axis=-1)
        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        image_labels = labels[:, :, None, None]
        image_labels = ops.repeat(
            image_labels, repeats=[self.image_dim * self.image_dim]
        )
        image_labels = ops.reshape(
            image_labels, (-1, self.image_dim, self.image_dim, self.num_classes)
        )

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batch_size = ops.shape(real_images)[0]
        random_latent_vectors = random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )
        random_vector_labels = ops.concatenate(
            [random_latent_vectors, labels], axis=1
        )

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator(random_vector_labels)

        combined_images = ops.concatenate(
            [generated_images, real_images], axis=0
        )

        # Assemble labels discriminating real from fake images.
        real_fake_labels = ops.concatenate(
            [
                ops.stack([ops.ones((batch_size, 1)), ops.expand_dims(age_labels, axis=-1), ops.expand_dims(gender_labels, axis=-1)], axis=-1),
                ops.stack([ops.zeros((batch_size, 1)),  ops.expand_dims(age_labels, axis=-1), ops.expand_dims(gender_labels, axis=-1)], axis=-1)
            ],
            axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(real_fake_labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )
        random_vector_labels = ops.concatenate(
            [random_latent_vectors, labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = ops.stack([ops.zeros((batch_size, 1)),  ops.expand_dims(age_labels, axis=-1), ops.expand_dims(gender_labels, axis=-1)], axis=-1)

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            predictions = self.discriminator(fake_images)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }
