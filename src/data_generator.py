import os
import cv2
# import imgaug as ia
# from imgaug import augmenters as iaa
import numpy as np

class DataGenerator(object):

	def __init__(self, folder_name="", image_size="", number_of_images="", data_split=0.8, augment=False):
		self.folder_name = folder_name
		self.image_size = image_size
		self.number_of_images = number_of_images
		self.data_split = data_split
		self.augment = augment

	def augment(self, image):
		pass

	"""Generate 3 splits of the dataset:
			1 image for testing, per object
			data_split of the data for training
			1 - data_split of the data for validation
	"""
	def generate(self):

		folder_name = self.folder_name

		x_train = []
		x_valid = []
		x_test = []

		folder_list = os.listdir(folder_name)
		for folder in folder_list:
			image_list = os.listdir(folder_name+folder)

			x_test.append(cv2.imread(folder_name+folder+"/"+image_list[0], 1))

			if self.augment:
				for image_name in image_list[1:int(self.data_split * self.number_of_images)]:
					image = cv2.imread(folder_name+folder+"/"+image_name, 1)
					x_train.append(image)
					for i in range(number_of_augmentations):
						x_train.append(augment(image))
			else: 
				for image_name in image_list[1:int(self.data_split * self.number_of_images)]:
					x_train.append(cv2.imread(folder_name+folder+"/"+image_name, 1))

			for image_name in image_list[int(self.data_split * self.number_of_images):]:
				x_valid.append(cv2.imread(folder_name+folder+"/"+image_name, 1))		

		x_train = np.array(x_train)
		x_valid = np.array(x_valid)
		x_test = np.array(x_test)
		x_train = x_train.astype('float32') / 255.
		x_valid = x_valid.astype('float32') / 255.
		x_test = x_test.astype('float32') / 255.

		# adapt this if using `channels_first` image data format
		x_train = np.reshape(x_train, (len(x_train), self.image_size, self.image_size, 3))
		x_valid = np.reshape(x_valid, (len(x_valid), self.image_size, self.image_size, 3))
		x_test = np.reshape(x_test, (len(x_test), self.image_size, self.image_size, 3))

		print("\nDATA_LOADED")
		print("Training: {0}".format(x_train.shape))
		print("Validation: {0}".format(x_valid.shape))
		print("Testing: {0}\n".format(x_test.shape))

		return x_train, x_valid, x_test