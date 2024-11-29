import wandb
from wandb.integration.keras import WandbMetricsLogger
from sklearn import model_selection
from keras import optimizers, ops

import argparse
from types import SimpleNamespace
import random
import csv
from pathlib import Path

from utils import load_face_data, GeneratorTestCallback
from data_generator import get_dataset_from_slices

from discriminator import get_discriminator
from generator import get_generator
from face_gan import FaceGAN


def train_and_evaluate_hyperparameters(hyperparameters, x, y, model_save_path, infer_previous_model,
                                       infer_finished_model):
    # Data information
    label_structure = ['age_output', 'gender_output']

    dataset = get_dataset_from_slices(x, y, hyperparameters)

    checkpoint_filepath = '/tmp/checkpoints/checkpoint.face.keras'

    num_classes = len(label_structure)
    num_channels = 3

    generator_in_channels = hyperparameters.latent_dim
    discriminator_in_channels = num_channels
    print(generator_in_channels, discriminator_in_channels)

    generator = get_generator(
        num_channels=generator_in_channels,
        image_dim=hyperparameters.image_dim,
    )

    def discriminator_loss(real_img, fake_img):
        real_loss = ops.mean(real_img)
        fake_loss = ops.mean(fake_img)
        return fake_loss - real_loss

    def generator_loss(y_pred):
        return -ops.mean(y_pred)

    discriminator = get_discriminator(
        input_shape=(hyperparameters.image_dim, hyperparameters.image_dim, discriminator_in_channels),
        dropout_rate=hyperparameters.dropout_rate,
    )

    model = FaceGAN(
        generator=generator,
        discriminator=discriminator,
        latent_dim=hyperparameters.latent_dim,
        discriminator_extra_steps=3,
    )

    model.compile(
        d_optimizer=optimizers.Adam(learning_rate=hyperparameters.learning_rate, beta_1=0.5, beta_2=0.9),
        g_optimizer=optimizers.Adam(learning_rate=hyperparameters.learning_rate, beta_1=0.5, beta_2=0.9),
        d_loss_fn=discriminator_loss,
        g_loss_fn=generator_loss,
        run_eagerly=False,
    )
    model.summary()

    model_callbacks = []

    def scheduler(epoch, lr):
        return float(lr * hyperparameters.learning_rate_factor)

    # model_callbacks.append(callbacks.LearningRateScheduler(scheduler))

    model_callbacks.append(GeneratorTestCallback(hyperparameters.latent_dim))

    try:
        if True:
            wandb.init(
                project="FACE-GAN",
                config={
                    "epochs": hyperparameters.epochs,
                    "batch_size": hyperparameters.batch_size,
                    "start_learning_rate": hyperparameters.learning_rate,
                    "learning_rate_factor": hyperparameters.learning_rate_factor,
                    "dropout_rate": hyperparameters.dropout_rate,
                    "image_dim": hyperparameters.image_dim,
                    "latent_dim": hyperparameters.latent_dim,
                })
            model_callbacks.append(WandbMetricsLogger())
    except Exception as e:
        print("No wandb callback added.")

    history = model.fit(x=dataset,
                        epochs=hyperparameters.epochs,
                        callbacks=model_callbacks)

    generator = model.generator
    generator.save(model_save_path / "FaceGAN-Generator.keras")

    discriminator = model.discriminator
    discriminator.save(model_save_path / "FaceGAN-Discriminator.keras")

    with open(model_save_path / 'training_history_facegan.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(history.history.keys())
        # Write data
        writer.writerows(zip(*history.history.values()))


def get_arg_parser():
    parser = argparse.ArgumentParser(description='Training hyperparameters')

    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')

    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')

    parser.add_argument('--image_dim', type=int, default=64,
                        help='Image dimensions (height, width)')

    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Generator input size (height, width)')

    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')

    parser.add_argument('--dropout-rate', type=float, default=0.25,
                        help='Dropout rate')

    parser.add_argument('--learning-rate-factor', type=float, default=0.9,
                        help='Learning rate decay factor')

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
    )

    print("\nRunning training with following hyperparameters:")
    print(f"\tEpochs: {hyperparameters.epochs}")
    print(f"\tBatch Size: {hyperparameters.batch_size}")
    print(f"\tImage Dimensions: {hyperparameters.image_dim}")
    print(f"\tLatent Dimensions: {hyperparameters.latent_dim}")
    print(f"\tLearning Rate: {hyperparameters.learning_rate}")
    print(f"\tDropout Rate: {hyperparameters.dropout_rate}")
    print(f"\tLearning Rate Factor: {hyperparameters.learning_rate_factor}")
    print()

    # Save information
    model_save_path = Path("saved_models")
    model_save_path.mkdir(parents=True, exist_ok=True)

    # Load data
    x, y = load_face_data(
        Path(__file__).parent / '../data/utk-face/UTKFace',
        Path(__file__).parent / '../data/ffhq/images256x256',
        with_ffhq=True,
        with_utk=False
    )

    train_and_evaluate_hyperparameters(hyperparameters, x, y, model_save_path, args.infer_previous_model,
                                       args.infer_finished_model)
