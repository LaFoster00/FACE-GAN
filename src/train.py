import wandb
from wandb.integration.keras import WandbMetricsLogger
from sklearn import model_selection
from keras import optimizers, ops
import keras
from functools import partial

import tensorflow as tf

import numpy as np

import argparse
from types import SimpleNamespace
import random
import csv
from pathlib import Path

from utils import load_face_data, GeneratorTestCallback, log2
from data_generator import get_dataset_from_slices

from face_gan import FaceGAN

import matplotlib.pyplot as plt


# we use different batch size for different resolution, so larger image size
# could fit into GPU memory. The keys is image resolution in log2
batch_sizes = {2: 16, 3: 16, 4: 16, 5: 16, 6: 16, 7: 8, 8: 4, 9: 2, 10: 1}
# We adjust the train step accordingly
train_step_ratio = {k: batch_sizes[2] / v for k, v in batch_sizes.items()}

def resize_image(res, image, label):
    # only downsampling, so use nearest neighbor that is faster to run
    image = tf.image.resize(
        image, (res, res), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return image, label


def create_dataloader(dataset, res):
    batch_size = batch_sizes[log2(res)]
    # NOTE: we unbatch the dataset so we can `batch()` it again with the `drop_remainder=True` option
    # since the model only supports a single batch size
    dl = dataset.map(partial(resize_image, res), num_parallel_calls=tf.data.AUTOTUNE).unbatch()
    dl = dl.shuffle(200).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).repeat()
    return dl

def plot_images(images, log2_res, fname=""):
    scales = {2: 0.5, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8}
    scale = scales[log2_res]

    grid_col = min(images.shape[0], int(32 // scale))
    grid_row = 1

    f, axarr = plt.subplots(
        grid_row, grid_col, figsize=(grid_col * scale, grid_row * scale)
    )

    for row in range(grid_row):
        ax = axarr if grid_row == 1 else axarr[row]
        for col in range(grid_col):
            ax[col].imshow(images[row * grid_col + col])
            ax[col].axis("off")
    plt.show()
    if fname:
        f.savefig(fname)


def train(
    style_gan,
    dataset,
    start_res=4,
    target_res=32,
    steps_per_epoch=5000,
    display_images=True,
):
    opt_cfg = {"learning_rate": 1e-3, "beta_1": 0.0, "beta_2": 0.99, "epsilon": 1e-8}

    val_batch_size = 16
    val_z = keras.random.normal((val_batch_size, style_gan.z_dim))
    val_noise = style_gan.generate_noise(val_batch_size)

    start_res_log2 = int(np.log2(start_res))
    target_res_log2 = int(np.log2(target_res))

    for res_log2 in range(start_res_log2, target_res_log2 + 1):
        res = 2 ** res_log2
        for phase in ["TRANSITION", "STABLE"]:
            if res == start_res and phase == "TRANSITION":
                continue

            train_dl = create_dataloader(dataset, res)

            steps = int(train_step_ratio[res_log2] * steps_per_epoch)

            style_gan.compile(
                d_optimizer=optimizers.Adam(**opt_cfg),
                g_optimizer=optimizers.Adam(**opt_cfg),
                loss_weights={"gradient_penalty": 10, "drift": 0.001},
                steps_per_epoch=steps,
                res=res,
                phase=phase,
                run_eagerly=False,
            )

            prefix = f"res_{res}x{res}_{style_gan.phase}"

            ckpt_cb = keras.callbacks.ModelCheckpoint(
                f"checkpoints/stylegan_{res}x{res}.weights.h5",
                save_weights_only=True,
                verbose=0,
            )
            print(phase)
            style_gan.fit(
                train_dl, epochs=1, steps_per_epoch=steps, callbacks=[ckpt_cb]
            )

            if display_images:
                images = style_gan({"z": val_z, "noise": val_noise, "alpha": 1.0})
                plot_images(images, res_log2)


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


    model = FaceGAN(
        z_dim=hyperparameters.latent_dim,
        target_res=hyperparameters.image_dim,
    )
    train(
        model,
        dataset
    )

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
