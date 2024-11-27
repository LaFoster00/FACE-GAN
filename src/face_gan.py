import argparse
import os
from types import SimpleNamespace

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
from discriminator import *




def train_and_evaluate_hyperparameters(hyperparameters, x, y, model_save_path, infer_previous_model, infer_finished_model):
    # Data information
    label_structure = ['age_output', 'gender_output']

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

    training_generator = DataGenerator(
        image_paths=x_train,
        labels=labels_train,
        label_structure=label_structure,
        batch_size=hyperparameters.batch_size,
        dim=hyperparameters.image_dim)

    val_generator = DataGenerator(
        image_paths=x_val,
        labels=labels_val,
        label_structure=label_structure,
        batch_size=hyperparameters.batch_size,
        dim=hyperparameters.image_dim)

    test_generator = DataGenerator(
        image_paths=x_test,
        labels=labels_test,
        label_structure=label_structure,
        batch_size=hyperparameters.batch_size,
        dim=hyperparameters.image_dim)

    checkpoint_filepath = '/tmp/checkpoints/checkpoint.face.keras'

    discriminator = get_discriminator(
        input_shape=(hyperparameters.image_dim, hyperparameters.image_dim, 3),
        dropout_rate=hyperparameters.dropout_rate,
        model=hyperparameters.model,
        freeze_base=hyperparameters.freeze_base
    )
    model = None
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

    parser.add_argument('--image_dim', type=int, default=128,
                        help='Image dimensions (height, width)')

    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Generator input size (height, width)')

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
        image_dim=args.image_dim,
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        learning_rate_factor=args.learning_rate_factor,
        model=args.model,
        freeze_base=args.freeze_base,
    )

    print("\nRunning training with following hyperparameters:")
    print(f"\tEpochs: {hyperparameters.epochs}")
    print(f"\tBatch Size: {hyperparameters.batch_size}")
    print(f"\tImage Dimensions: {hyperparameters.image_dim}")
    print(f"\tLatent Dimensions: {hyperparameters.latent_dim}")
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
