from keras import layers, models, ops, backend
import keras
import numpy as np

from layers import WeightedAdd, PixelNorm, ToRGB
from src.utils import lerp_clip, cset, lerp

from utils import number_features


def get_generator(
        latents_shape=(64, ),  # First input: Latent vectors [minibatch, latent_size].
        labels_shape=(2,),  # Second input: Labels [minibatch, label_size].
        num_channels=3,  # Number of output color channels. Overridden based on dataset.
        resolution=32,  # Output resolution. Overridden based on dataset.
        label_size=0,  # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
        latent_size=None,  # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
        fmap_base=8192,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        normalize_latents=True,  # Normalize latent vectors before feeding them to the network?
        use_pixelnorm=True,  # Enable pixelwise feature vector normalization?
        pixelnorm_epsilon=1e-8,  # Constant epsilon for pixelwise feature vector normalization.
        structure=None,  # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
        is_template_graph=False,  # True = template graph constructed by the Network class, False = actual evaluation.
):
    latents_in = layers.Input(shape=latents_shape, name='latents_in')
    labels_in = layers.Input(shape=labels_shape, name='labels_in')
    resolution_log2 = int(np.log2(resolution))

    def nf(log2res):
        return number_features(log2res, fmap_base, fmap_decay, fmap_max)

    def pn(x):
        return PixelNorm(epsilon=pixelnorm_epsilon)(x) if use_pixelnorm else x

    def upscale2d(x, factor=2):
        return layers.UpSampling2D(size=(factor, factor), interpolation='nearest')(x)

    def torgb(x, log2res):
        return ToRGB(log2res, num_channels)(x)

    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    if latent_size is None: latent_size = nf(0)

    combo_in = ops.cast(ops.concatenate([latents_in, labels_in], axis=1), backend.floatx())
    lod_in = ops.cast(keras.Variable(name='lod', initializer=np.float32(0.0), trainable=False), backend.floatx())

    keras.Variable()
    def generator_block(x, log2res):
        if log2res == 2:  # 4x4
            if normalize_latents: x = PixelNorm()(x)
            x = layers.Dense(nf(log2res - 1) * 16)(x)
            x = layers.Reshape(target_shape=(-1, 4, 4, nf(log2res-1)))(x)
            x = layers.LeakyReLU(negative_slope=0.2)(x)
            x = pn(x)
            x = layers.Conv2D(
                filters=nf(log2res-1),
                kernel_size=(3, 3),
                strides=(1,1),
                padding='same',
                data_format='channels_last')(x)
            x = layers.LeakyReLU(negative_slope=0.2)(x)
            x = pn(x)
        else: # 8x8 and up
            x = upscale2d(x)
            x = layers.Conv2D(
                filters=nf(log2res - 1),
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                data_format='channels_last')(x)
            x = layers.LeakyReLU(negative_slope=0.2)(x)
            x = pn(x)
            x = layers.Conv2D(
                filters=nf(log2res - 1),
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                data_format='channels_last')(x)
            x = layers.LeakyReLU(negative_slope=0.2)(x)
            x = pn(x)
        return x

    if structure == 'linear':
        x = generator_block(combo_in, 2)
        images_out = torgb(x, 2)
        for res in range(3, resolution_log2 + 1):
            lod = resolution_log2 - res
            x = generator_block(x, res)
            img = torgb(x, res)
            images_out = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(images_out)
            images_out = lerp_clip(img, images_out, lod_in - lod)

    if structure == 'recursive':
        def grow(x, res, lod):
            y = generator_block(x, res)
            img = lambda: upscale2d(torgb(y, res), factor=2**lod)
            if res > 2: img = cset(img, (lod_in > lod), lambda: upscale2d(lerp(torgb(y, res), upscale2d(torgb(x, res - 1)), lod_in -lod), 2**lod))
            if lod > 0: img = cset(img, (lod_in < lod), lambda: grow(y, res + 1, lod))
            return img()
        images_out = grow(combo_in, 2, resolution_log2 - 2)

    images_out = layers.Identity(name='images_out')(images_out)
    model = models.Model(inputs=[latents_in, labels_in], outputs=images_out)
    return model
