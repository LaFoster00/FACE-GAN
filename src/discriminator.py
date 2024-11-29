from keras import models, layers, models, ops, backend
import keras

from layers import WeightedAdd
import numpy as np

from layers import FromRGB, PixelNorm

from utils import number_features, lerp_clip, cset, lerp


def get_discriminator_model(
    images_in,                          # Input: Images [minibatch, height, width, channel].
    num_channels        = 1,            # Number of input color channels. Overridden based on dataset.
    resolution          = 32,           # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
):
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(log2res):
        return number_features(log2res, fmap_base, fmap_decay, fmap_max)
    def downscale2D(x, factor=2):
        # TODO check if strides is correct it might need to be (1,1)
        return layers.AveragePooling2D(pool_size=(factor, factor), strides=(factor, factor), padding='valid',data_format='channel_first')(x)
    def fromrgb(x, log2res):
        return FromRGB(log2res, fmap_base, fmap_decay, fmap_max)(x)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'

    lod_in = ops.cast(keras.Variable('lod', initializer=np.float32(0.0), trainable=False), backend.floatx())

    def discriminator_block(x, log2res):
        if log2res >= 3: # 8x8 and up
            x = layers.Conv2D(
                filters=nf(log2res - 1),
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                data_format='channels_last')(x)
            x = layers.LeakyReLU(negative_slope=0.2)(x)
            x = layers.Conv2D(
                filters=nf(log2res - 2),
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                data_format='channels_last')(x)
            x = layers.LeakyReLU(negative_slope=0.2)(x)
            x = downscale2D(x)
        else: # 4x4
            x = layers.Conv2D(
                filters=nf(log2res - 1),
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                data_format='channels_last')(x)
            x = layers.LeakyReLU(negative_slope=0.2)(x)
            x = layers.Conv2D(
                filters=nf(log2res - 2),
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                data_format='channels_last')(x)
            x = layers.LeakyReLU(negative_slope=0.2)(x)
            x = layers.Dense(1+label_size)(x)
        return x

    # Linear structure: simple but inefficient
    if structure == 'linear':
        img = images_in
        x = fromrgb(img, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            lod = resolution_log2 - res
            x = discriminator_block(x, res)
            img = downscale2D(img)
            y = fromrgb(img, res - 1)
            x = lerp_clip(x, y, lod_in - lod)
        combo_out = discriminator_block(x, 2)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(res, lod):
            x = lambda: fromrgb(downscale2D(img, 2**lod), res)
            if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
            x = discriminator_block(x(), res); y = lambda: x
            if res > 2: y = cset(y, (lod_in > lod), lambda: lerp(x, fromrgb(downscale2D(images_in, 2**(lod+1)), res - 1), lod_in - lod))
            return y()
        combo_out = grow(2, resolution_log2 - 2)

    scores_out = layers.Identity(name='score_out')(combo_out[:, :1])
    labels_out = layers.Identity(name='labels_out')(combo_out[:, 1:])
    return scores_out, labels_out




