from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import cv2

class VanillaConvAE(object):


	def __init__(self):
		self.model_name = "vanilla_conv_ae"
		self.latent_size = 8

		self.history = None
		self.encoder = None
		self.decoder = None
		self.autoencoder = None


	def build(self):
		# define model
		input_img = Input(shape=(100, 100, 3))  # adapt this if using `channels_first` image data format
		x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img) # (100, 100)
		x = MaxPooling2D((2, 2), padding='same')(x) # (50, 50)
		x = Conv2D(8, (3, 3), activation='relu', padding='same')(x) # (50, 50)
		x = MaxPooling2D((2, 2), padding='same')(x) # (25, 25)
		x = Conv2D(8, (4, 4), activation='relu')(x) # (22, 22)
		x = MaxPooling2D((2, 2), padding='same')(x) # (11, 11)
		x = Conv2D(8, (4, 4), activation='relu')(x) # (8, 8)
		x = MaxPooling2D((2, 2), padding='same')(x) # (4, 4)
		encoded = Conv2D(8, (4, 4), activation='relu')(x) # (1, 1)

		# at this point the representation is (1, 1, 8) i.e. 128-dimensional

		x = UpSampling2D((4,4))(encoded) # (4,4)
		x = Conv2D(8, (3, 3), activation='relu', padding='same')(x) # (4, 4)
		x = UpSampling2D((2, 2))(x) # (8, 8)
		x = Conv2D(8, (3, 3), activation='relu', padding='same')(x) # (8, 8)
		x = UpSampling2D((2, 2))(x) # (16, 16)
		x = Conv2D(16, (3, 3), activation='relu')(x) # (14, 14)
		x = UpSampling2D((2, 2))(x) # (28, 28)
		x = Conv2D(16, (4, 4), activation='relu')(x) # (25, 25)
		x = UpSampling2D((2, 2))(x) # (50, 50)
		x = Conv2D(16, (3, 3), activation='relu', padding='same')(x) # (50, 50)
		x = UpSampling2D((2, 2))(x) # (100, 100)
		decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x) # (100, 100)

		self.autoencoder = Model(input_img, decoded)
		self.encoder = Model(input_img, encoded)


	def fit(self, x=None, y=None, batch_size=None, epochs=None, validation_data=None, shuffle=None):
		self.history = self.autoencoder.fit(x=x,
											y=y,
											epochs=epochs,
											batch_size=batch_size,
											shuffle=shuffle,
											validation_data=validation_data)


	def compile(self, optimizer=None, loss=None, metrics=None):
		self.autoencoder.compile(optimizer=optimizer,
								 loss=loss)


	def predict(self, data):
		return self.autoencoder.predict(data)


	def encode(self, data):
		return self.encoder.predict(data)


	def decode(self, data):
		return self.decoder.predict(data)


	def plot_learning_curve(self):
	    #plot learnign curves
	    plt.plot(self.history.history['loss'])
	    plt.plot(self.history.history['val_loss'])
	    plt.title('model loss')
	    plt.ylabel('loss')
	    plt.xlabel('epoch')
	    plt.legend(['train', 'valid'], loc='upper right')
	    plt.savefig("results/"+self.model_name+"/curve")


	def inspect_result(self, encoded_imgs, decoded_imgs, x_test):
	    number_of_objects = 12
	    image_shape = decoded_imgs[0].shape
	    plt.figure(figsize=(20, 6))

	    for i in range(number_of_objects):
	        # display original
	        ax = plt.subplot(2, number_of_objects, i + 1)
	        test_image = x_test[i].reshape(image_shape)
	        plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
	        plt.gray()
	        ax.get_xaxis().set_visible(False)
	        ax.get_yaxis().set_visible(False)

	        # display reconstruction
	        ax = plt.subplot(2, number_of_objects, i + number_of_objects + 1)
	        decoded_image = decoded_imgs[i].reshape(image_shape)
	        plt.imshow(cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB))
	        plt.gray()
	        ax.get_xaxis().set_visible(False)
	        ax.get_yaxis().set_visible(False)
	    plt.savefig("results/"+self.model_name+"/reconstructions")
	    plt.close()

	def inspect_latent(self, encoded_imgs):
		number_of_objects = 12
		# plot vectors
		plt.figure(figsize=(20, 8))
		for i in range(number_of_objects):
			ax = plt.subplot(1, number_of_objects, i + 1)
			plt.imshow(encoded_imgs[i].reshape(1, self.latent_size).T)
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
		plt.savefig("results/"+self.model_name+"/latent_vectors")
		plt.close()


class BetaConvVAE(VanillaConvAE):


	def __init__(self, beta=1, number_to_augment=0):
		self.model_name = "beta_conv_vae_beta" + str(beta) + "_augment_" + str(number_to_augment) 

		if not os.path.exists("results/" + self.model_name):
			os.mkdir("results/" + self.model_name)

		self.latent_size = 2

		self.beta = beta
		self.history = None
		self.encoder = None
		self.decoder = None
		self.autoencoder = None


	def build(self):
		# define model
		# image format is channels last - (batch_size, x, y, no_filters)
		input_img = 		Input(shape=(100, 100, 3))  # adapt this if using `channels_first` image data format
		conv_0_encoded = 	Conv2D(16, (3, 3), activation='relu', padding='same')(input_img) # (100, 100)
		pool_0_encoded = 	MaxPooling2D((2, 2), padding='same')(conv_0_encoded) # (50, 50)
		conv_1_encoded = 	Conv2D(8, (3, 3), activation='relu', padding='same')(pool_0_encoded) # (50, 50)
		pool_1_encoded = 	MaxPooling2D((2, 2), padding='same')(conv_1_encoded) # (25, 25)
		conv_2_encoded = 	Conv2D(8, (4, 4), activation='relu')(pool_1_encoded) # (22, 22)
		pool_2_encoded = 	MaxPooling2D((2, 2), padding='same')(conv_2_encoded) # (11, 11)
		conv_3_encoded = 	Conv2D(8, (4, 4), activation='relu')(pool_2_encoded) # (8, 8)
		reshaped_encoded = 	Flatten()(conv_3_encoded) # (1,64)
		dense_0_encoded = 	Dense(8)(reshaped_encoded) # (1,8)

		z_mean = Dense(2)(dense_0_encoded)
		z_log_var = Dense(2)(dense_0_encoded)

		def sampling(args):
		    z_mean, z_log_var = args
		    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 2), mean=0., stddev=1.0)
		    return z_mean + K.exp(z_log_var) * epsilon

		z = Lambda(sampling, output_shape=(2,))([z_mean, z_log_var]) # (1,2)

		# define layers
		decoder_dense_0 = 		Dense(8) # (1,8)
		decoder_dense_1 = 		Dense(8 * 8) # (1,64)
		decoder_reshaped = 		Reshape([8, 8, 1]) # (8,8)
		decoder_deconv_0 = 		Conv2D(8, (3, 3), activation='relu', padding='same') # (8, 8)
		decoder_up_0 = 			UpSampling2D((2, 2)) # (16, 16)
		decoder_deconv_1 = 		Conv2D(16, (3, 3), activation='relu') # (14, 14)
		decoder_up_1 = 			UpSampling2D((2, 2)) # (28, 28)
		decoder_deconv_2 = 		Conv2D(16, (4, 4), activation='relu') # (25, 25)
		decoder_up_2 = 			UpSampling2D((2, 2)) # (50, 50)
		decoder_deconv_3 = 		Conv2D(16, (3, 3), activation='relu', padding='same') # (50, 50)
		decoder_up_3 = 			UpSampling2D((2, 2)) # (100, 100)
		decoder_output_img = 	Conv2D(3, (3, 3), activation='sigmoid', padding='same') # (100, 100)

		# instantiate layers for training
		dense_0_decoded = 	decoder_dense_0(z) # (1,8)
		dense_1_decoded =	decoder_dense_1(dense_0_decoded) # (1,64)
		reshaped_decoded = 	decoder_reshaped(dense_1_decoded) # (8,8)
		deconv_0_decoded = 	decoder_deconv_0(reshaped_decoded) # (8, 8)
		up_0_decoded = 		decoder_up_0(deconv_0_decoded) # (16, 16)
		deconv_1_decoded = 	decoder_deconv_1(up_0_decoded) # (14, 14)
		up_1_decoded = 		decoder_up_1(deconv_1_decoded) # (28, 28)
		deconv_2_decoded = 	decoder_deconv_2(up_1_decoded) # (25, 25)
		up_2_decoded = 		decoder_up_2(deconv_2_decoded) # (50, 50)
		deconv_3_decoded = 	decoder_deconv_3(up_2_decoded) # (50, 50)
		up_3_decoded = 		decoder_up_3(deconv_3_decoded) # (100, 100)
		output_img = 		decoder_output_img(up_3_decoded) # (100, 100)

		# instantiate layers for test-time generation from latent space samples
		latent = 				Input(shape=(self.latent_size,))
		_dense_0_decoded = 		decoder_dense_0(latent) # (1,8)
		_dense_1_decoded =		decoder_dense_1(_dense_0_decoded) # (1,64)
		_reshaped_decoded = 	decoder_reshaped(_dense_1_decoded) # (8,8)
		_deconv_0_decoded = 	decoder_deconv_0(_reshaped_decoded) # (8, 8)
		_up_0_decoded = 		decoder_up_0(_deconv_0_decoded) # (16, 16)
		_deconv_1_decoded = 	decoder_deconv_1(_up_0_decoded) # (14, 14)
		_up_1_decoded = 		decoder_up_1(_deconv_1_decoded) # (28, 28)
		_deconv_2_decoded = 	decoder_deconv_2(_up_1_decoded) # (25, 25)
		_up_2_decoded = 		decoder_up_2(_deconv_2_decoded) # (50, 50)
		_deconv_3_decoded = 	decoder_deconv_3(_up_2_decoded) # (50, 50)
		_up_3_decoded = 		decoder_up_3(_deconv_3_decoded) # (100, 100)
		_output_img = 			decoder_output_img(_up_3_decoded) # (100, 100)
		
		# define the 3 models
		self.autoencoder =	Model(input_img, output_img)
		self.encoder = 		Model(input_img, z_mean)
		self.decoder = 		Model(latent, _output_img)

		xent_loss = 100 * 100 * 3 * metrics.binary_crossentropy(K.flatten(input_img), K.flatten(output_img))
		kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
		vae_loss = K.mean(xent_loss + self.beta*kl_loss)
		self.autoencoder.add_loss(vae_loss)

	# sample from a normal distribution and generate images from the samples
	def generate(self):

		n = 10
		image_size = 100

		figure = np.zeros((image_size * n, image_size * n, 3))
		grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
		grid_y = norm.ppf(np.linspace(0.05, 0.95, n))


		for i, yi in enumerate(grid_x):
		    for j, xi in enumerate(grid_y):
		        z_sample = np.array([[xi, yi]])
		        # z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
		        x_decoded = self.decoder.predict(z_sample)
		        image_sample = x_decoded.reshape(image_size, image_size, 3)
		        figure[i * image_size: (i + 1) * image_size,
		               j * image_size: (j + 1) * image_size,
		               :] = image_sample

		figure = (figure*255).astype(np.uint8)

		cv2.imwrite("results/"+self.model_name+"/latent_samples.png", figure)

	# project a set of images to the learned latent space and calculate statistics
	# over the values of the resultant latent values
	def inspect_latent_distribution(self, data):

		latent_vars = self.encoder.predict(data)
		print(latent_vars)
		mean_vector = np.mean(latent_vars, axis=0).tolist()
		cov = np.cov(latent_vars.T)
		cov = np.diag(np.diag(cov))
		sigmas = np.diag(np.sqrt(cov)).tolist()

		print(mean_vector)
		print(sigmas)

		colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'orange', 'maroon', 'lime', 'salmon', 'crimson', 'azure', 'coral']
		bins = np.array(range(-50,100)) / 100.

		for i, mean in enumerate(mean_vector):
			y = mlab.normpdf( bins, mean, sigmas[i])
			l = plt.plot(bins, y, colors[i], linewidth=3, label="Latent dim " + str(i))
			
		plt.legend()
		plt.title("Distribution of the latent space")
		# plt.axis([0, 1, 0, 40])
		plt.grid(True)
		plt.savefig("results/"+self.model_name+"/latent_value_distributions")
		plt.close()