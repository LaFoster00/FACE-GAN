import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'

import wandb
from wandb.integration.keras import WandbMetricsLogger
from sklearn import model_selection
from keras import optimizers, losses, callbacks, utils, random
import keras
import numpy as np

import argparse
from types import SimpleNamespace
import csv
from pathlib import Path

from utils import load_face_data, GeneratorTestCallback, plot_images
from data_generator import get_dataset_from_slices

from face_gan import StyleGAN

# we use different batch size for different resolution, so larger image size
# could fit into GPU memory. The keys is image resolution in log2
batch_sizes = {2: 16, 3: 16, 4: 16, 5: 16, 6: 16, 7: 8, 8: 4, 9: 2, 10: 1}
# We adjust the train step accordingly
train_step_ratio = {k: batch_sizes[2] / v for k, v in batch_sizes.items()}


def train(
        model: StyleGAN,
        start_res,
        target_res,
        hyperparameters,
        dataset,
        display_images=True,
        callbacks=[],
        checkpoint_path=Path(__file__).parent / '../saved_models/checkpoints',
        model_save_path=Path(__file__).parent / '../saved_models',
):
    opt_cfg = {"learning_rate": hyperparameters.learning_rate, "beta_1": 0.0, "beta_2": 0.99, "epsilon": 1e-8}

    val_batch_size = 16
    val_z = random.normal((val_batch_size, model.z_dim))
    val_noise = model.generate_noise(val_batch_size)

    start_res_log2 = int(np.log2(start_res))
    target_res_log2 = int(np.log2(target_res))

    for res_log2 in range(start_res_log2, target_res_log2 + 1):
        res = 2 ** res_log2
        for phase in ["TRANSITION", "STABLE"]:
            if res == start_res and phase == "TRANSITION":
                continue

            steps = int(train_step_ratio[res_log2] * hyperparameters.steps_per_epoch)

            model.compile(
                d_optimizer= optimizers.Adam(**opt_cfg),
                g_optimizer= optimizers.Adam(**opt_cfg),
                loss_weights={"gradient_penalty": 10, "drift": 0.001},
                steps_per_epoch=steps,
                res=res,
                phase=phase,
                run_eagerly=True,
            )

            model.summary()

            callbacks.append( keras.callbacks.ModelCheckpoint(
                checkpoint_path / f"facegan_{res}x{res}.weights.h5",
                save_weights_only=True,
                verbose=0,
            ))

            print(phase)
            model.fit(x=dataset,
                      epochs=5,
                      steps_per_epoch=steps,
                      callbacks=callbacks
            )

            generator = model.generator
            generator.save(model_save_path / "FaceGAN-Generator.keras")

            discriminator = model.discriminator
            discriminator.save(model_save_path / "FaceGAN-Discriminator.keras")

            if display_images:
                images = model({"z": val_z, "noise": val_noise, "alpha": 1.0})
                plot_images(images, res_log2)


def train_and_evaluate_hyperparameters(hyperparameters, x, y, model_save_path):
    # Data information
    label_structure = ['age_output', 'gender_output']

    dataset = get_dataset_from_slices(x, y, hyperparameters)

    checkpoint_path = Path(__file__).parent / "../saved_models/checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    START_RES = 4
    TARGET_RES = hyperparameters.image_dim

    model_callbacks = []
    model_callbacks.append(GeneratorTestCallback(hyperparameters.latent_dim))

    try:
        if True:
            wandb.init(
                project="FACE-GAN",
                config={
                    "epochs": hyperparameters.epochs,
                    "steps_per_epoch": hyperparameters.steps_per_epoch,
                    "batch_size": hyperparameters.batch_size,
                    "start_learning_rate": hyperparameters.learning_rate,
                    "learning_rate_factor": hyperparameters.learning_rate_factor,
                    "image_dim": hyperparameters.image_dim,
                    "latent_dim": hyperparameters.latent_dim,
                })
            model_callbacks.append(WandbMetricsLogger())
    except Exception as e:
        print("No wandb callback added.")

    model = StyleGAN(start_res=START_RES, target_res=TARGET_RES)

    train(model,
          start_res=START_RES,
          target_res=TARGET_RES,
          hyperparameters=hyperparameters,
          dataset=dataset,
          display_images=True,
          callbacks=model_callbacks,
          checkpoint_path=checkpoint_path,
          model_save_path=model_save_path)



def get_arg_parser():
    parser = argparse.ArgumentParser(description='Training hyperparameters')

    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')

    parser.add_argument('--steps-per-epoch', type=int, default=5000,
                        help='Number of steps per epoch')

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

    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    parser.print_usage()
    args = parser.parse_args()

    # Convert to SimpleNamespace if needed
    hyperparameters = SimpleNamespace(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        image_dim=args.image_dim,
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
        learning_rate_factor=args.learning_rate_factor,
    )

    print("\nRunning training with following hyperparameters:")
    print(f"\tEpochs: {hyperparameters.epochs}")
    print(f"\tSteps per epoch: {hyperparameters.steps_per_epoch}")
    print(f"\tBatch Size: {hyperparameters.batch_size}")
    print(f"\tImage Dimensions: {hyperparameters.image_dim}")
    print(f"\tLatent Dimensions: {hyperparameters.latent_dim}")
    print(f"\tLearning Rate: {hyperparameters.learning_rate}")
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

    train_and_evaluate_hyperparameters(hyperparameters, x, y, model_save_path)
