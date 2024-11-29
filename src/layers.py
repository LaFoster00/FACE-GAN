import numpy as np
from keras import layers, ops
import keras

from utils import number_features


# Custom Keras layer to randomly convert an image to grayscale during training
@keras.saving.register_keras_serializable()
class RandomGrayscale(layers.Layer):
    def __init__(self, probability=0.5, **kwargs):
        super().__init__(**kwargs)
        self.probability = probability

    def call(self, inputs, training=True):
        """
        In training mode, randomly convert input images to grayscale with a given probability.
        This helps the model generalize better.
        """
        if training:
            if np.random.random() > self.probability:
                # Convert the image to grayscale using the luminance formula
                grayscale = ops.dot(inputs[..., :3], [0.2989, 0.5870, 0.1140])  # Grayscale formula
                grayscale = ops.expand_dims(grayscale, axis=-1)
                # Concatenate the grayscale values back to 3 channels (RGB) to maintain dimensions
                inputs = ops.concatenate((grayscale, grayscale, grayscale), axis=-1)
        return inputs

    def get_config(self):
        """
        Serialize the configuration of the RandomGrayscale layer so it can be saved.
        """
        config = super().get_config()
        config.update({"probability": self.probability})
        return config

# Preprocessing pipeline that applies random data augmentations
def preprocessing_pipeline(inputs, preprocessing):
    """
    Applies a sequence of random augmentations to the input images:
    - Random Zoom
    - Random Rotation
    - Random Horizontal Flip
    - Random Brightness
    - Random Contrast
    - Random Grayscale conversion
    """
    #x = layers.RandomZoom(0.2)(inputs)
    x = layers.RandomRotation(0.1)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomBrightness(0.1)(x)
    x = layers.RandomContrast(0.1)(x)
    x = RandomGrayscale(probability=0.5)(x)
    x = preprocessing(x)
    return x

# Used to face in the new layer when increasing the convolution size
def fade_in(a, b, alpha):
    return ((1 - alpha) * a) + (alpha * b)

# Weighted add off two tensors of same shape
class WeightedAdd(layers.Add):
    def __init__(self, alpha=0.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = keras.Variable(alpha, name='ws_alpha')

    #Output a weighted addition of inputs
    def _merge_function(self, inputs):
        # only supports a weighted sum of two inputs
        assert(len(inputs) == 2)
        return fade_in(inputs[0], inputs[1], alpha=self.alpha)

class PixelNorm(layers.Layer):
    def __init__(self, epsilon=1e-8,**kwargs):
        super().__init__(**kwargs)
        self.epsilon = keras.Variable(epsilon)

    def call(self, inputs):
        return inputs * ops.rsqrt(ops.mean(ops.square(inputs), axis=1, keepdims=True) + self.epsilon)

class ToRGB(layers.Conv2D):
    def __init__(self, log2res, num_channels, **kwargs):
        super().__init__(
            filters=num_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            name=f"ToRGB_{log2res}",
            **kwargs)

class FromRGB(layers.Conv2D):
    def __init__(self, log2res, fmap_base=8192, fmap_decay=1.0, fmap_max=512, **kwargs):
        super().__init__(
            filters=number_features(log2res, fmap_base, fmap_decay, fmap_max),
            kernel_size=(1, 1),
            strides=(1,1),
            name=f"FromRGB_{log2res}",
            **kwargs)
