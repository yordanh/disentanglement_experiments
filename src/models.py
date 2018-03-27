from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Flatten, Reshape
from keras.models import Model
from keras.models import model_from_json
from keras import backend as K
from keras import metrics
import numpy as np
import os
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import cv2

class AutoEncoder(object):
	def __init__(self, data_generator=None):
		self.data_generator = data_generator
		self.results_folder = ""
		self.models_folder = ""

		self.history = None
		self.encoder = None
		self.decoder = None
		self.autoencoder = None


	def build(self):
		pass


	def fit(self, x=None, y=None, batch_size=None, epochs=None, validation_data=None, shuffle=None):

		callbacks_list = []
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
		callbacks_list.append(early_stopping)

		self.history = self.autoencoder.fit(x=x,
											y=y,
											epochs=epochs,
											batch_size=batch_size,
											callbacks=callbacks_list,
											shuffle=shuffle,
											validation_data=validation_data)


	def compile(self, optimizer=None, loss=None, metrics=None):
		self.autoencoder.compile(optimizer=optimizer,
								 loss=loss)
	

	def save_model(self):
		# Serialize model to JSON
		if self.encoder != None:
			encoder_json = self.encoder.to_json()
			with open(self.models_folder + "/encoder.json", "w") as json_file:
				json_file.write(encoder_json)
		   	# Serialize weights to HDF5
    		self.encoder.save_weights(self.models_folder + "/encoder.h5")
    		print("Saved ENCODER model to disk")

		if self.decoder:
			decoder_json = self.decoder.to_json()
			with open(self.models_folder + "/decoder.json", "w") as json_file:
				json_file.write(decoder_json)
			# Serialize weights to HDF5
			self.decoder.save_weights(self.models_folder + "/decoder.h5")
			print("Saved DECODER model to disk")

		full_model_json = self.autoencoder.to_json()
		with open(self.models_folder + "/full.json", "w") as json_file:
			json_file.write(full_model_json)
		# Serialize weights to HDF5
		self.autoencoder.save_weights(self.models_folder + "/full.h5")
		
		
		print("Saved FULL model to disk")

	def load_model(self):

		try:
			json_file = open(self.models_folder + '/encoder.json', 'r')
			loaded_model_json = json_file.read()
			json_file.close()
			self.encoder = model_from_json(loaded_model_json)
			
			# Load weights into new model
			self.encoder.load_weights(self.models_folder + "/encoder.h5")
			print("Loaded ENCODER model from disk")
		except:
			print("No ENCODER model found!")

		try:
			json_file = open(self.models_folder + '/decoder.json', 'r')
			loaded_model_json = json_file.read()
			json_file.close()
			self.decoder = model_from_json(loaded_model_json)
			
			# Load weights into new model
			self.decoder.load_weights(self.models_folder + "/decoder.h5")
			print("Loaded DECODER model from disk")
		except:
			print("No DECODER model found!")

		try:
			#  Load json and create model
			json_file = open(self.models_folder + '/full.json', 'r')
			loaded_model_json = json_file.read()
			json_file.close()
			self.autoencoder = model_from_json(loaded_model_json)
			
			# Load weights into new model
			self.autoencoder.load_weights(self.models_folder + "/full.h5")
			print("Loaded FULL model from disk")
		except:
			print("No FULL model found!")


	def predict(self, data):
		return self.autoencoder.predict(data)


	def encode(self, data):
		return self.encoder.predict(data)


	def decode(self, data):
		return self.decoder.predict(data)


	# plot the trainign and validation losses as functions of time
	def plot_learning_curve(self):
	    #plot learnign curves
	    plt.plot(self.history.history['loss'])
	    plt.plot(self.history.history['val_loss'])
	    plt.title('model loss')
	    plt.ylabel('loss')
	    plt.xlabel('epoch')
	    plt.legend(['train', 'valid'], loc='upper right')
	    plt.savefig(self.results_folder + "/curve")
	    plt.close()


	# plot the original and reconstructed images
	def plot_real_vs_reconstruct(self, decoded_imgs, x_test):
	    n = self.data_generator.number_of_objects
	    image_shape = decoded_imgs[0].shape
	    plt.figure(figsize=(20, 6))

	    # display original vs reconstructed images
	    for i in range(n):
	        ax = plt.subplot(2, n, i + 1)
	        test_image = x_test[i].reshape(image_shape)
	        plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
	        plt.gray()
	        ax.get_xaxis().set_visible(False)
	        ax.get_yaxis().set_visible(False)

	        # display reconstruction
	        ax = plt.subplot(2, n, i + n + 1)
	        decoded_image = decoded_imgs[i].reshape(image_shape)
	        plt.imshow(cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB))
	        plt.gray()
	        ax.get_xaxis().set_visible(False)
	        ax.get_yaxis().set_visible(False)
	    plt.savefig(self.results_folder + "/reconstructions")
	    plt.close()


	# plot latent vectors for a set of encoded images
	def plot_latent_vectors(self, encoded_imgs):
		n = self.data_generator.number_of_objects
		# plot vectors
		plt.figure(figsize=(20, 8))
		for i in range(n):
			ax = plt.subplot(1, n, i + 1)
			plt.imshow(encoded_imgs[i].reshape(1, self.latent_size).T)
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
		plt.savefig(self.results_folder + "/latent_vectors")
		plt.close()

	# sample from a normal distribution and generate images from the samples
	def plot_sampled_images(self, samples_per_dimension=10, image_size=100):

		figure = np.zeros((image_size * samples_per_dimension, image_size * samples_per_dimension, 3))
		grid_x = norm.ppf(np.linspace(0.05, 0.95, samples_per_dimension))
		grid_y = norm.ppf(np.linspace(0.05, 0.95, samples_per_dimension))


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

		cv2.imwrite(self.results_folder + "/latent_samples.png", figure)

	# project a set of images to the learned latent space and calculate statistics
	# over the values of the resultant latent values
	def plot_latent_distribution(self, data):

		latent_vars = self.encoder.predict(data)
		mean_vector = np.mean(latent_vars, axis=0).tolist()
		cov = np.cov(latent_vars.T)
		cov = np.diag(np.diag(cov))
		sigmas = np.diag(np.sqrt(cov)).tolist()

		print("Latent {0}".format(latent_vars))
		print("Means {0}".format(mean_vector))
		print("Sigmas {0}".format(sigmas))

		colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'orange', 'maroon', 'lime', 'salmon', 'crimson', 'azure', 'coral']
		bins = np.array(range(-50,100)) / 10.

		for i, mean in enumerate(mean_vector):
			y = mlab.normpdf( bins, mean, sigmas[i])
			l = plt.plot(bins, y, colors[i], linewidth=3, label="Latent dim " + str(i))
			
		plt.legend()
		plt.title("Distribution of the latent space")
		plt.grid(True)
		plt.savefig(self.results_folder + "/latent_value_distributions")
		plt.close()

class VanillaConvAE(AutoEncoder):


	def __init__(self, data_generator=None, latent_size=8):
		self.data_generator = data_generator
		self.model_name = "vanilla_conv_ae_augment_" + str(self.data_generator.number_to_augment)
		self.results_folder = "results/" + self.data_generator.data_name + "/" + self.model_name
		self.models_folder = "models/" + self.data_generator.data_name + "/" + self.model_name

		if not os.path.exists(self.results_folder):
			os.makedirs(self.results_folder)

		if not os.path.exists(self.models_folder):
			os.makedirs(self.models_folder)

		self.latent_size = latent_size

		self.history = None
		self.encoder = None
		self.decoder = None
		self.autoencoder = None

	# define model
	# image format is channels last - (batch_size, x, y, no_filters)
	def build(self):
		input_img = Input(shape=(100, 100, 3))  # adapt this if using `channels_first` image data format
		x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img) # (100, 100)
		x = MaxPooling2D((2, 2), padding='same')(x) # (50, 50)
		x = Conv2D(8, (3, 3), activation='relu', padding='same')(x) # (50, 50)
		x = MaxPooling2D((2, 2), padding='same')(x) # (25, 25)
		x = Conv2D(8, (4, 4), activation='relu')(x) # (22, 22)
		x = MaxPooling2D((2, 2), padding='same')(x) # (11, 11)
		x = Conv2D(8, (4, 4), activation='relu')(x) # (8, 8)
		reshaped_encoded = Flatten()(x)

		latent = Dense(8)(reshaped_encoded) # (1,8)

		upsampled_decoded = Dense(8 * 8)(latent) # (1,64)
		reshaped_decoded = Reshape([8, 8, 1])(upsampled_decoded) # (8,8)
		x = Conv2D(8, (3, 3), activation='relu', padding='same')(reshaped_decoded) # (8, 8)
		x = UpSampling2D((2, 2))(x) # (16, 16)
		x = Conv2D(16, (3, 3), activation='relu')(x) # (14, 14)
		x = UpSampling2D((2, 2))(x) # (28, 28)
		x = Conv2D(16, (4, 4), activation='relu')(x) # (25, 25)
		x = UpSampling2D((2, 2))(x) # (50, 50)
		x = Conv2D(16, (3, 3), activation='relu', padding='same')(x) # (50, 50)
		x = UpSampling2D((2, 2))(x) # (100, 100)
		output_img = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x) # (100, 100)

		self.autoencoder = Model(input_img, output_img)
		self.encoder = Model(input_img, latent)


class BetaConvVAE(AutoEncoder):


	def __init__(self, beta=1, data_generator=None, latent_size=2):
		self.data_generator = data_generator
		self.model_name = "beta_conv_vae_beta" + str(beta) + "_augment_" + str(self.data_generator.number_to_augment) 
		self.results_folder = "results/" + self.data_generator.data_name + "/" + self.model_name
		self.models_folder = "models/" + self.data_generator.data_name + "/" + self.model_name

		if not os.path.exists(self.results_folder):
			os.makedirs(self.results_folder)

		if not os.path.exists(self.models_folder):
			os.makedirs(self.models_folder)

		self.latent_size = latent_size

		self.beta = beta
		self.history = None
		self.encoder = None
		self.decoder = None
		self.autoencoder = None


	# define model
	# image format is channels last - (batch_size, x, y, no_filters)
	def build(self):
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