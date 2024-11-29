from keras import layers, models, ops

def upsample_block(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(1, 1),
    up_size=(2, 2),
    padding="same",
    use_bn=False,
    use_bias=True,
    use_dropout=False,
    drop_value=0.3,
):
    x = layers.UpSampling2D(up_size)(x)
    x = layers.Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)

    if use_bn:
        x = layers.BatchNormalization()(x)

    if activation:
        x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x

def get_generator(num_channels, image_dim):
    input = layers.Input(shape=(num_channels,))

    num_conv = 3
    upscale_dim = int(image_dim / (2**num_conv))

    x = layers.Dense(upscale_dim * upscale_dim * num_channels)(input)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Reshape((upscale_dim, upscale_dim, num_channels))(x)
    x = upsample_block(
        x,
        128,
        layers.LeakyReLU(0.2),
        strides=(1, 1),
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x,
        64,
        layers.LeakyReLU(0.2),
        strides=(1, 1),
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x, 3, layers.Activation("sigmoid"), strides=(1, 1), use_bias=False, use_bn=True
    )

    model = models.Model(inputs=input, outputs=x)
    model.name = "generator"
    model.summary()

    return model