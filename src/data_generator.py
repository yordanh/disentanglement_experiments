import os
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

class DataAugmenter(object):
	def __init__(self):
		self.seq = iaa.SomeOf((2, 7), [
				iaa.Fliplr(0.5), # horizontal flips

				iaa.Flipud(0.5), # horizontal flips

				# iaa.Crop(percent=(0, 0.1)), # random crops

				# Small gaussian blur with random sigma between 0 and 0.5.
				# But we only blur about 50% of all images.
				iaa.Sometimes(0.5,
				    iaa.GaussianBlur(sigma=(0, 0.5))
				),

				# Strengthen or weaken the contrast in each image.
				iaa.ContrastNormalization((0.75, 1.5)),

				# Add gaussian noise.
				# For 50% of all images, we sample the noise once per pixel.
				# For the other 50% of all images, we sample the noise per pixel AND
				# channel. This can change the color (not only brightness) of the
				# pixels.
				iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),

				# Make some images brighter and some darker.
				# In 20% of all cases, we sample the multiplier once per channel,
				# which can end up changing the color of the images.
				iaa.Multiply((0.8, 1.2), per_channel=0.2),

				# Apply affine transformations to each image.
				# Scale/zoom them, translate/move them, rotate them and shear them.
				iaa.Affine(
				    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
				    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
				    rotate=(-5, 5)
				)
				], random_order=True) # apply augmenters in random order

	def augment(self, images):
		return self.seq.augment_images(images)

class DataGenerator(object):

	def __init__(self, folder_name="", image_size=0, number_of_images=0, data_split=0.8, number_to_augment=0):
		self.folder_name = folder_name
		self.image_size = image_size
		self.number_of_images = number_of_images
		self.data_split = data_split
		self.number_to_augment = number_to_augment
		if self.number_to_augment != 0:
			self.augmenter = DataAugmenter()

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

			print("Processing folder {0}/{1} with {2} images".format(folder_list.index(folder), len(folder_list), len(image_list)))

			x_test.append(cv2.imread(folder_name+folder+"/"+image_list[0], 1))

			if self.number_to_augment != 0:
				for image_name in image_list[1:int(self.data_split * self.number_of_images)]:
					image = cv2.imread(folder_name+folder+"/"+image_name, 1)
					x_train.append(image)

					images_for_augmentation = np.tile(image, (self.number_to_augment, 1, 1))
					shape = image.shape
					images_for_augmentation = images_for_augmentation.reshape(self.number_to_augment, shape[0], shape[1], shape[2])
					augmented = self.augmenter.augment(images_for_augmentation)
					x_train += augmented.tolist()

					# for image_a in augmented:
					# 	cv2.imshow("original", image)
					# 	cv2.imshow("augmented", image_a)
					# 	cv2.waitKey(0)
					# exit()
			else: 
				for image_name in image_list[1:int(self.data_split * self.number_of_images)]:
					x_train.append(cv2.imread(folder_name+folder+"/"+image_name, 1))

			for image_name in image_list[int(self.data_split * self.number_of_images):]:
				x_valid.append(cv2.imread(folder_name+folder+"/"+image_name, 1))		

			print("Training size: {0} images".format(len(x_train)))

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