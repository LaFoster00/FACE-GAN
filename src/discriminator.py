from keras import models, layers, applications, metrics, losses, optimizers, callbacks, saving, ops, utils, backend
import keras
from keras.src.layers import LeakyReLU
from keras.src.ops import BinaryCrossentropy

from layers import preprocessing_pipeline

"""
This module contains a multi-task deep learning model for face detection, age estimation, and gender classification. 
The model uses a pre-trained EfficientNetB0 architecture as a base model, and is extended with custom output layers for each task.

Key Features:
- **Face Detection**: A binary classification task to detect the presence of a face in an image.
- **Age Estimation**: A regression task to predict the age of a person in the image.
- **Gender Classification**: A multi-class classification task to predict the gender of the person in the image.
"""


# Custom loss and metric functions for the age and gender tasks.
@keras.saving.register_keras_serializable()
def age_loss_fn(real_true, age_true, age_pred):
    """
    Custom loss function for age prediction. It computes Mean Squared Error only for valid age values (age < 200).

    Args:
    - y_true: Ground truth labels (age values).
    - y_pred: Predicted age values.

    Returns:
    - loss: Mean squared error between true and predicted ages for valid entries.
    """
    age_pred = age_pred * ops.cast(ops.equal(real_true, 1), age_pred.dtype)  # Mask invalid age values (>= 200)
    age_true = age_true * ops.cast(ops.equal(real_true, 1), age_true.dtype)
    return losses.mean_squared_error(age_true, age_pred)


@keras.saving.register_keras_serializable()
def age_metric(y_true, y_pred):
    """
    Custom metric function for age prediction. It computes Mean Absolute Error only for valid age values (age < 200).

    Args:
    - y_true: Ground truth labels (age values).
    - y_pred: Predicted age values.

    Returns:
    - metric: Mean absolute error between true and predicted ages for valid entries.
    """
    mask = ops.less(y_true, 200)
    mask_pred = ops.expand_dims(mask, axis=-1)

    y_pred = ops.where(mask_pred, y_pred, ops.zeros_like(y_pred))
    y_true = ops.where(mask, y_true, ops.zeros_like(y_true))
    return metrics.mean_absolute_error(y_true, y_pred)


@keras.saving.register_keras_serializable()
def gender_loss_fn(real_true, gender_true, gender_pred):
    """
    Custom loss function for gender prediction. It computes Binary Cross-Entropy only for valid gender labels.

    Args:
    - y_true: Ground truth labels (gender values).
    - y_pred: Predicted gender probabilities.

    Returns:
    - loss: Binary cross-entropy between true and predicted gender labels.
    """
    gender_pred = gender_pred * ops.cast(ops.equal(real_true, 1), gender_pred.dtype)  # Mask invalid gender values (>= 2)
    gender_true = gender_true * ops.cast(ops.equal(real_true, 1), gender_true.dtype)
    return losses.binary_crossentropy(gender_true, gender_pred)


@keras.saving.register_keras_serializable()
def gender_metric(y_true, y_pred):
    """
    Custom metric function for gender prediction. It computes Binary Accuracy only for valid gender labels.

    Args:
    - y_true: Ground truth labels (gender values).
    - y_pred: Predicted gender probabilities.

    Returns:
    - metric: Binary accuracy between true and predicted gender labels.
    """

    mask = ops.less(y_true, 2)
    mask_pred = ops.expand_dims(mask, axis=-1)

    y_pred = ops.where(mask_pred, y_pred, ops.zeros_like(y_pred))
    y_true = ops.where(mask, y_true, ops.zeros_like(y_true))
    return metrics.binary_accuracy(y_true, y_pred)

def discriminator_preprocessing(inputs):
    return inputs / 255.0

def get_discriminator(
        input_shape,
        dropout_rate=0.25,
        model='efficientnet-b0',
        freeze_base=True
):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(input_shape[0], kernel_size=4, strides=2, padding="same")(inputs)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Conv2D(input_shape[0] * 2, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Conv2D(input_shape[0] * 2, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1, activation="sigmoid")(x)

    # Combine all branches into a final model
    model = models.Model(inputs=inputs,
                         outputs=x)
    model.name = 'discriminator'
    model.summary()

    return model


def compile_discriminator(discriminator, learning_rate):
    # Compile the model with respective loss functions and metrics
    discriminator.compile(
        run_eagerly=False,
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss={
            'age_output': age_loss_fn,
            'gender_output': gender_loss_fn,
        },
        metrics={
            'age_output': age_metric,
            'gender_output': gender_metric,
        })

def discriminator_loss(y_true, y_pred):
    return losses.binary_crossentropy(y_true, y_pred)

