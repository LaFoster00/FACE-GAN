from keras import layers, models, optimizers, utils
import keras

from layers import WeightedAdd

# add a discriminator block
def add_discriminator_block(old_model, n_input_layers=3):
	# get shape of existing model
	in_shape = old_model.input.shape
	# define new input shape as double the size
	input_shape = (in_shape[-2] *2, in_shape[-2] *2, in_shape[-1])
	in_image = layers.Input(shape=input_shape)
	# define new input processing layer
	d = layers.Conv2D(64, (1,1), padding='same', kernel_initializer='he_normal')(in_image)
	d = layers.LeakyReLU(alpha=0.2)(d)
	# define new block
	d = layers.Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(d)
	d = layers.BatchNormalization()(d)
	d = layers.LeakyReLU(alpha=0.2)(d)
	d = layers.Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(d)
	d = layers.BatchNormalization()(d)
	d = layers.LeakyReLU(alpha=0.2)(d)
	d = layers.AveragePooling2D(pool_size=(2,2))(d)
	block_new = d
	# skip the input, 1x1 and activation for the old model
	for i in range(n_input_layers, len(old_model.layers)):
		d = old_model.layers[i](d)
	# define straight-through model
	model1 = models.Model(in_image, d)
	# compile model
	model1.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
	# downsample the new larger image
	downsample = layers.AveragePooling2D(pool_size=(2,2))(in_image)
	# connect old input processing to downsampled new input
	block_old = old_model.layers[1](downsample)
	block_old = old_model.layers[2](block_old)
	# fade in output of old model input layer with new input
	d = WeightedAdd()([block_old, block_new])
	# skip the input, 1x1 and activation for the old model
	for i in range(n_input_layers, len(old_model.layers)):
		d = old_model.layers[i](d)
	# define straight-through model
	model2 = models.Model(in_image, d)
	# compile model
	model2.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
	return [model1, model2]


# define the discriminator models for each image resolution
def define_discriminator(n_blocks, input_shape=(4,4,3)):
	model_list = list()
	# base model input
	in_image = layers.Input(shape=input_shape)
	# conv 1x1
	d = layers.Conv2D(64, (1,1), padding='same', kernel_initializer='he_normal')(in_image)
	d = layers.LeakyReLU(alpha=0.2)(d)
	# conv 3x3 (output block)
	d = layers.Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(d)
	d = layers.BatchNormalization()(d)
	d = layers.LeakyReLU(alpha=0.2)(d)
	# conv 4x4
	d = layers.Conv2D(128, (4,4), padding='same', kernel_initializer='he_normal')(d)
	d = layers.BatchNormalization()(d)
	d = layers.LeakyReLU(alpha=0.2)(d)
	# dense output layer
	d = layers.Flatten()(d)
	out_class = layers.Dense(1)(d)
	# define model
	model = keras.models.Model(in_image, out_class)
	# compile model
	model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
	# store model
	model_list.append([model, model])
	# create submodels
	for i in range(1, n_blocks):
		# get prior model without the fade-on
		old_model = model_list[i - 1][0]
		# create new model for next resolution
		models = add_discriminator_block(old_model)
		# store model
		model_list.append(models)
	return model_list

if __name__ == '__main__':
	# define models
	discriminators = define_discriminator(3)
	# spot check
	m = discriminators[2][1]
	m.summary()
	utils.plot_model(m, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)