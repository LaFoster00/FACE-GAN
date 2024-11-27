import tensorflow as tf


def get_dataset_from_slices(x, y, hyperparameters, num_parallel_calls=tf.data.AUTOTUNE, prefetch_size=tf.data.AUTOTUNE):
    def load_and_preprocess_image(image_path, label):
        # Read image
        image = tf.io.read_file(filename=image_path)
        # Decode image
        image = tf.cond(
            tf.image.is_jpeg(image),
            lambda: tf.image.decode_jpeg(contents=image, channels=3),
            lambda: tf.image.decode_png(contents=image, channels=3))

        # Resize image
        image = tf.image.resize(images=image, size=(hyperparameters.image_dim, hyperparameters.image_dim))
        # Normalize pixel values
        image = image / 255.0
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=num_parallel_calls)
    dataset = (
        dataset.shuffle(buffer_size=len(x))
        .batch(hyperparameters.batch_size)
        .prefetch(buffer_size=prefetch_size)
    )
    return dataset