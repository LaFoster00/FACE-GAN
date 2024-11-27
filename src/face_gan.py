import argparse
import os
from types import SimpleNamespace

import numpy as np
from matplotlib import pyplot

import wandb
from keras import models, layers, applications, metrics, losses, optimizers, callbacks, saving, ops, utils, backend
from utils import load_face_data
from data_generator import DataGenerator
import matplotlib.pyplot as plt
from sklearn import model_selection
import csv
from wandb.integration.keras import WandbMetricsLogger
from pathlib import Path
import keras
import random

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
def age_loss_fn(y_true, y_pred):
    """
    Custom loss function for age prediction. It computes Mean Squared Error only for valid age values (age < 200).

    Args:
    - y_true: Ground truth labels (age values).
    - y_pred: Predicted age values.

    Returns:
    - loss: Mean squared error between true and predicted ages for valid entries.
    """
    y_pred = y_pred * ops.cast(ops.less(y_true, 200), y_pred.dtype)  # Mask invalid age values (>= 200)
    y_true = y_true * ops.cast(ops.less(y_true, 200), y_true.dtype)
    return losses.mean_squared_error(y_true, y_pred)


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
def gender_loss_fn(y_true, y_pred):
    """
    Custom loss function for gender prediction. It computes Binary Cross-Entropy only for valid gender labels.

    Args:
    - y_true: Ground truth labels (gender values).
    - y_pred: Predicted gender probabilities.

    Returns:
    - loss: Binary cross-entropy between true and predicted gender labels.
    """
    y_pred = y_pred * ops.cast(ops.less(y_true, 2), y_pred.dtype)  # Mask invalid gender values (>= 2)
    y_true = y_true * ops.cast(ops.less(y_true, 2), y_true.dtype)
    return losses.binary_crossentropy(y_true, y_pred)


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


def FaceIdentifier(
        input_shape=(256, 256, 3),
        dropout_rate=0.25,
        learning_rate=3e-4,
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

    # Define face output branch (binary classification)
    face_output = layers.Dropout(rate=dropout_rate, name='face_dropout')(x)
    face_output = layers.Dense(1, activation='sigmoid', name='face_output')(face_output)

    # Define age output branch (regression)
    age_output = layers.Dense(2024, activation='relu', name='age_1')(x)
    age_output = layers.BatchNormalization()(age_output)
    age_output = layers.Dense(1024, activation='relu', name='age_2')(age_output)
    age_output = layers.BatchNormalization()(age_output)
    age_output = layers.Dropout(rate=dropout_rate, name='age_dropout')(age_output)
    age_output = layers.Dense(1, activation='relu', name='age_output')(age_output)

    # Define gender output branch (multi-class classification)
    gender_output = layers.Dense(2024, activation='relu', name='gender_1')(x)
    gender_output = layers.BatchNormalization()(gender_output)
    gender_output = layers.Dense(1024, activation='relu', name='gender_2')(gender_output)
    gender_output = layers.BatchNormalization()(gender_output)
    gender_output = layers.Dropout(rate=dropout_rate, name='gender_dropout')(gender_output)
    gender_output = layers.Dense(1, activation='sigmoid', name='gender_output')(gender_output)

    # Combine all branches into a final model
    model = models.Model(inputs=inputs, outputs={'face_output': face_output,
                                                 'age_output': age_output,
                                                 'gender_output': gender_output})

    # Compile the model with respective loss functions and metrics
    model.compile(
        run_eagerly=False,
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss={
            'face_output': losses.BinaryCrossentropy(),
            'age_output': age_loss_fn,
            'gender_output': gender_loss_fn,
        },
        metrics={
            'face_output': metrics.BinaryAccuracy(),
            'age_output': age_metric,
            'gender_output': gender_metric,
        })

    # Plot the model architecture for visualization
    utils.plot_model(model)

    return model


def infer_images(images, model, show=True):
    """
    Performs inference on a list of images using the given model.

    Args:
    - images: List of images for inference.
    - model: Pre-trained Keras model for inference.
    - show: Whether to display the image during inference (default: True).

    Returns:
    - None
    """
    results = []
    for image in images:
         results.append(infer_image(image, model, show))
    return results


def infer_image(image, model, show=True):
    """
    Performs inference on a single image using the given model and prints the result.

    Args:
    - image: Single image for inference.
    - model: Pre-trained Keras model for inference.
    - show: Whether to display the image during inference (default: True).

    Returns:
    - label: The result label with face, age, and gender prediction.
    """
    if show:
        plt.imshow(image)
        plt.show()

    predictions = model.predict(ops.expand_dims(image, 0))
    score_face = float(predictions['face_output'][0][0])
    score_age = round(predictions['age_output'][0][0])
    score_gender = float(predictions['gender_output'][0][0])

    label = f"This image contains a face with {100 * score_face:.2f}% certainty."
    print(label)

    if score_face > 0.5:
        additional_label = f"The person has gender {'male' if score_gender <= 0.5 else 'female'} and is {score_age} years old."
        print(additional_label)
        label += '\n' + additional_label

    return label


def train_and_evaluate_hyperparameters(hyperparameters, x, y, model_save_path, infer_previous_model, infer_finished_model):
    # Data information
    label_structure = ['face_output', 'age_output', 'gender_output']

    # Step 1: Split data into training (80%) and test+validation (20%) sets
    x_train, x_temp, labels_train, labels_temp = model_selection.train_test_split(x,
                                                                                  y,
                                                                                  test_size=0.2,
                                                                                  random_state=random.randint(0, 20000))

    # Step 2: Split the remaining 20% data into validation (10%) and test (10%) sets
    x_val, x_test, labels_val, labels_test = model_selection.train_test_split(x_temp,
                                                                              labels_temp,
                                                                              test_size=0.5,
                                                                              random_state=random.randint(0, 20000))

    if infer_previous_model and os.path.exists(model_save_path / "Face.keras"):
        try:
            model = saving.load_model(model_save_path / "Face.keras")
            infer_images(
                DataGenerator(x, y, label_structure, batch_size=8, for_fitting=False, prefetch_batches=1, dim=hyperparameters.dim)[0],
                model)
        except Exception as e:
            print(e)

    training_generator = DataGenerator(
        image_paths=x_train,
        labels=labels_train,
        label_structure=label_structure,
        batch_size=hyperparameters.batch_size,
        dim=hyperparameters.dim)

    val_generator = DataGenerator(
        image_paths=x_val,
        labels=labels_val,
        label_structure=label_structure,
        batch_size=hyperparameters.batch_size,
        dim=hyperparameters.dim)

    test_generator = DataGenerator(
        image_paths=x_test,
        labels=labels_test,
        label_structure=label_structure,
        batch_size=hyperparameters.batch_size,
        dim=hyperparameters.dim)

    checkpoint_filepath = '/tmp/checkpoints/checkpoint.face.keras'

    model = FaceIdentifier(
        input_shape=(*hyperparameters.dim, 3),
        dropout_rate=hyperparameters.dropout_rate,
        learning_rate=hyperparameters.learning_rate,
        model=hyperparameters.model,
        freeze_base=hyperparameters.freeze_base
    )
    model.summary()
    # if os.path.exists(checkpoint_filepath):
    # model.load_weights(checkpoint_filepath)

    model_callbacks = []

    model_callbacks.append(callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    ))

    model_callbacks.append(callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=5,
        restore_best_weights=True,
        mode="min"
    ))

    def scheduler(epoch, lr):
        return float(lr * hyperparameters.learning_rate_factor)

    model_callbacks.append(callbacks.LearningRateScheduler(scheduler))

    try:
        wandb.init(
            project="FACE-GAN",
            config={
                "epochs": hyperparameters.epochs,
                "batch_size": hyperparameters.batch_size,
                "start_learning_rate": hyperparameters.learning_rate,
                "learning_rate_factor": hyperparameters.learning_rate_factor,
                "dropout": hyperparameters.dropout_rate,
                "base_model": hyperparameters.model,
                "freeze_base": hyperparameters.freeze_base,
            })
        model_callbacks.append(WandbMetricsLogger())
    except Exception as e:
        print("No wandb callback added.")

    history = model.fit(x=training_generator,
                        validation_data=val_generator,
                        epochs=hyperparameters.epochs,
                        callbacks=model_callbacks)

    result = model.evaluate(x=test_generator)
    print(result)

    model.save(model_save_path / "Face.keras")

    if infer_finished_model:
        model = saving.load_model(model_save_path / "Face.keras")
        infer_images(DataGenerator(x, y, label_structure, batch_size=8, for_fitting=False, dim=hyperparameters.dim)[0], model)

    with open(model_save_path / 'training_history_dropout_face.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(history.history.keys())
        # Write data
        writer.writerows(zip(*history.history.values()))


def get_arg_parser():
    parser = argparse.ArgumentParser(description='Training hyperparameters')

    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')

    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')

    parser.add_argument('--dim', nargs=2, type=int, default=[256, 256],
                        help='Input dimensions (height, width)')

    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')

    parser.add_argument('--dropout-rate', type=float, default=0.25,
                        help='Dropout rate')

    parser.add_argument('--learning-rate-factor', type=float, default=0.9,
                        help='Learning rate decay factor')

    parser.add_argument('--model', type=str, default='efficientnet-b0',
                        choices=['efficientnet-b0', 'efficientnet-b4',
                                 'resnet50', 'resnet101', 'inception', 'mobilenet'],
                        help='Model architecture')

    parser.add_argument('--freeze-base', type=bool, default=True,
                        help='Freeze the base model for training.')

    parser.add_argument('--infer-previous-model', action='store_true', default=False,
                        help='Run inference on 8 of images using the previously trained model, before training.')

    parser.add_argument('--infer-finished-model', action='store_true', default=False,
                        help='Run inference on 8 of images using the newly trained model, after training.')

    return parser



if __name__ == '__main__':
    parser = get_arg_parser()
    parser.print_usage()
    args = parser.parse_args()


    # Convert to SimpleNamespace if needed
    hyperparameters = SimpleNamespace(
        epochs=args.epochs,
        batch_size=args.batch_size,
        dim=tuple(args.dim),
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        learning_rate_factor=args.learning_rate_factor,
        model=args.model,
        freeze_base=args.freeze_base,
    )

    print("\nRunning training with following hyperparameters:")
    print(f"\tEpochs: {hyperparameters.epochs}")
    print(f"\tBatch Size: {hyperparameters.batch_size}")
    print(f"\tDimensions: {hyperparameters.dim}")
    print(f"\tLearning Rate: {hyperparameters.learning_rate}")
    print(f"\tDropout Rate: {hyperparameters.dropout_rate}")
    print(f"\tLearning Rate Factor: {hyperparameters.learning_rate_factor}")
    print(f"\tModel: {hyperparameters.model}")
    print(f"\tFreeze Base Model: {hyperparameters.freeze_base}")
    print()

    # Save information
    model_save_path = Path("saved_models")
    model_save_path.mkdir(parents=True, exist_ok=True)

    # Load data
    x, y = load_face_data(
        Path(__file__).parent / '../data/utk-face/UTKFace',
        Path(__file__).parent / '../data/ffhq/images256x256',
        with_ffhq=False,
    )

    train_and_evaluate_hyperparameters(hyperparameters, x, y, model_save_path, args.infer_previous_model, args.infer_finished_model)
