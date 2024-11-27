from keras import models, layers, applications, metrics, losses, optimizers, callbacks, saving, ops, utils, backend
import keras

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


def get_discriminator(
        input_shape=(256, 256, 3),
        dropout_rate=0.25,
        model='efficientnet-b0',
        freeze_base=True
):
    """
    Defines and compiles a multi-task model for face detection, age estimation, and gender classification.

    Args:
    - input_shape: Shape of the input images (default: (128, 128, 3)).
    - dropout_rate: Dropout rate for regularization (default: 0.25).

    Returns:
    - model: Compiled Keras model for face identification, age, and gender prediction.
    """
    inputs = layers.Input(shape=input_shape)

    # Load pre-trained model with frozen weights
    basemodel = None
    preprocessing = None

    if model == 'efficientnet-b0':
        basemodel = applications.efficientnet.EfficientNetB0(weights='imagenet', include_top=False)
        preprocessing = applications.efficientnet.preprocess_input
    elif model == 'efficientnet-b4':
        basemodel = applications.efficientnet.EfficientNetB4(weights='imagenet', include_top=False)
        preprocessing = applications.efficientnet.preprocess_input
    elif model == 'resnet50':
        basemodel = applications.ResNet50(weights='imagenet', include_top=False)
        preprocessing = applications.resnet.preprocess_input
    elif model == 'resnet101':
        basemodel = applications.ResNet101(weights='imagenet', include_top=False)
        preprocessing = applications.resnet.preprocess_input
    elif model == 'inception':
        basemodel = applications.InceptionV3(weights='imagenet', include_top=False)
        preprocessing = applications.inception_v3.preprocess_input
    elif model == 'mobilenet':
        basemodel = applications.MobileNetV2(weights='imagenet', include_top=False)
        preprocessing = applications.mobilenet_v2.preprocess_input
    else:
        raise ValueError("Unsupported model type specified.")

    if basemodel is None:
        raise Exception('Base model not defined.')
    basemodel.trainable = not freeze_base

    # Apply preprocessing pipeline (augmentations, etc.)
    x = preprocessing_pipeline(inputs, preprocessing)
    x = basemodel(x)

    # Add global average pooling and batch normalization
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    # Define image real branch (binary)
    real_output = layers.Dropout(dropout_rate)(x)
    real_output = layers.Dense(1, activation='sigmoid')(real_output)

    # Define age output branch (regression)
    age_output = layers.Dense(256, activation='relu', name='age_1')(x)
    age_output = layers.BatchNormalization()(age_output)
    age_output = layers.Dense(128, activation='relu', name='age_2')(age_output)
    age_output = layers.BatchNormalization()(age_output)
    age_output = layers.Dropout(rate=dropout_rate, name='age_dropout')(age_output)
    age_output = layers.Dense(1, activation='relu', name='age_output')(age_output)

    # Define gender output branch (multi-class classification)
    gender_output = layers.Dense(256, activation='relu', name='gender_1')(x)
    gender_output = layers.BatchNormalization()(gender_output)
    gender_output = layers.Dense(128, activation='relu', name='gender_2')(gender_output)
    gender_output = layers.BatchNormalization()(gender_output)
    gender_output = layers.Dropout(rate=dropout_rate, name='gender_dropout')(gender_output)
    gender_output = layers.Dense(1, activation='sigmoid', name='gender_output')(gender_output)

    # Combine all branches into a final model
    model = models.Model(inputs=inputs,
                         outputs=
                         {'real_output': real_output,
                          'age_output': age_output,
                          'gender_output': gender_output})
    model.name = 'discriminator'

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
    real_true = y_true[:, 0, 0]
    age_true = y_true[:, 0, 1]
    gender_true = y_true[:, 0, 2]

    real_pred = y_pred['real_output'][:, 0]
    age_pred = y_pred['age_output'][:, 0]
    gender_pred = y_pred['gender_output'][:, 0]

    real_loss = ops.mean(losses.binary_crossentropy(real_true, real_pred))
    age_loss = ops.mean(age_loss_fn(real_true, age_true, age_pred))
    gender_loss = ops.mean(gender_loss_fn(real_true, gender_true, gender_pred))

    return ops.sum([real_loss, age_loss, gender_loss])

