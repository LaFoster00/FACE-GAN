import numpy as np
from keras import layers, ops
import keras

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


def fade_in(alpha, a, b):
    return alpha * a + (1.0 - alpha) * b


def wasserstein_loss(y_true, y_pred):
    return -ops.mean(y_true * y_pred)


def pixel_norm(x, epsilon=1e-8):
    return x / ops.sqrt(ops.mean(x ** 2, axis=-1, keepdims=True) + epsilon)


def minibatch_std(input_tensor, epsilon=1e-8):
    n, h, w, c = ops.shape(input_tensor)
    if not n is None:
        group_size = ops.minimum(4, n)
    else:
        group_size = n

    x = ops.reshape(input_tensor, (group_size, -1, h, w, c))
    group_mean, group_var = ops.nn.moments(x, axes=[0], keepdims=False)
    group_std = ops.sqrt(group_var + epsilon)
    avg_std = ops.mean(group_std, axis=[1, 2, 3], keepdims=True)
    x = ops.tile(avg_std, [group_size, h, w, 1])
    return ops.concatenate([input_tensor, x], axis=-1)


class EqualizedConv(layers.Layer):
    def __init__(self, out_channels, kernel=3, gain=2, **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.out_channels = out_channels
        self.gain = gain
        self.pad = kernel != 1

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
        self.w = self.add_weight(
            shape=[self.kernel, self.kernel, self.in_channels, self.out_channels],
            initializer=initializer,
            trainable=True,
            name="kernel",
        )
        self.b = self.add_weight(
            shape=(self.out_channels,), initializer="zeros", trainable=True, name="bias"
        )
        fan_in = self.kernel * self.kernel * self.in_channels
        self.scale = ops.sqrt(self.gain / fan_in)

    def call(self, inputs):
        if self.pad:
            x = ops.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        else:
            x = inputs
        output = (
            ops.conv(x, self.scale * self.w, strides=1, padding="VALID")+ self.b
        )
        return output


class EqualizedDense(layers.Layer):
    def __init__(self, units, gain=2, learning_rate_multiplier=1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.gain = gain
        self.learning_rate_multiplier = learning_rate_multiplier

    def build(self, input_shape):
        self.in_channels = int(input_shape[-1])  # Convert to int for static shape
        initializer = keras.initializers.RandomNormal(
            mean=0.0, stddev=1.0 / self.learning_rate_multiplier
        )
        self.w = self.add_weight(
            shape=(self.in_channels, self.units),
            initializer=initializer,
            trainable=True,
            name="kernel",
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="zeros", trainable=True, name="bias"
        )
        # Store gain/fan_in as a constant to avoid TensorFlow graph issues
        fan_in = float(self.in_channels)
        self.scale = (self.gain / fan_in) ** 0.5

    def call(self, inputs):
        # Dynamically compute scaled weights
        scaled_w = self.w * self.scale
        # Perform matrix multiplication and bias addition
        output = ops.matmul(inputs, scaled_w) + self.b
        return output * self.learning_rate_multiplier


class AddNoise(layers.Layer):
    def build(self, input_shape):
        n, h, w, c = input_shape[0]
        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
        self.b = self.add_weight(
            shape=[1, 1, 1, c], initializer=initializer, trainable=True, name="kernel"
        )

    def call(self, inputs):
        x, noise = inputs
        output = x + self.b * noise
        return output


class AdaIN(layers.Layer):
    def __init__(self, gain=1, **kwargs):
        super().__init__(**kwargs)
        self.gain = gain

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        w_shape = input_shapes[1]

        self.w_channels = w_shape[-1]
        self.x_channels = x_shape[-1]

        self.dense_1 = EqualizedDense(self.x_channels, gain=1)
        self.dense_2 = EqualizedDense(self.x_channels, gain=1)

    def call(self, inputs):
        x, w = inputs
        ys = ops.reshape(self.dense_1(w), (-1, 1, 1, self.x_channels))
        yb = ops.reshape(self.dense_2(w), (-1, 1, 1, self.x_channels))
        return ys * x + yb
