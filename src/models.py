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


	    plt.figure(figsize=(20, 8))
	    for i in range(number_of_objects):
	        ax = plt.subplot(1, number_of_objects, i + 1)
	        plt.imshow(encoded_imgs[i].reshape(1, self.latent_size).T)
	        plt.gray()
	        ax.get_xaxis().set_visible(False)
	        ax.get_yaxis().set_visible(False)
	    plt.savefig("results/"+self.model_name+"/latent_vectors")
