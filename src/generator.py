from keras import layers, models, ops

def get_generator(num_channels, image_dim):
    input = layers.Input(shape=(num_channels,))

    num_conv = 4
    upscale_dim = int(image_dim / (2**num_conv))

    # We want to generate 128 + num_classes coefficients to reshape into a
    # 7x7x(128 + num_classes) map.
    x = layers.Dense(upscale_dim * upscale_dim * num_channels)(input)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Reshape((upscale_dim, upscale_dim, num_channels))(x)
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Conv2D(3, (upscale_dim, upscale_dim), padding="same", activation="sigmoid")(x)

    model = models.Model(inputs=input, outputs=x)
    model.name = "generator"

    return model