import numpy as np
import cv2
import math
import h5py
import numpy
import argparse
import os
import shutil
import itertools
import copy

parser = argparse.ArgumentParser(description='Process the dSprited dataset.')
parser.add_argument('--image_size', default=100, type=int, help='Width and height of the square patch in px.')
parser.add_argument('--cutoff', default=1, type=int, help='Cutoff number - max number of images per class extracted')

data = np.load("/home/yordan/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")

def prep_dir(folder_name):
	print("Prepring " + folder_name)
	if os.path.exists(folder_name):
		print("Cleaning " + folder_name)
		
		map(lambda object_folder : shutil.rmtree(folder_name + object_folder), os.listdir(folder_name))

		print(folder_name + " has been cleaned!")
	else:
		os.makedirs(folder_name)


def extract(folder_name=None, latent_spec=None, cutoff=None, image_size=None, bgr_color=numpy.array([255, 255, 255]), verbose=False):

	print("Extracting images for " + folder_name)
	indecies = []
	for i, c in enumerate(data['latents_classes']):
		if (c[1] in latent_spec['shape'] and
		    c[2] in latent_spec['scale'] and 
		    c[3] in latent_spec['orientation'] and
		    c[4] in latent_spec['x'] and
		    c[5] in latent_spec['y']):
		    indecies.append(i)

	images = numpy.take(data['imgs'], indecies, axis=0)

	for i, image in enumerate(images):
		if i > cutoff:
			break

		image = cv2.resize(image, (image_size, image_size))
		image = numpy.tile(image.reshape(image_size,image_size,1), (1, 1, 3)) * bgr_color
		print(image.shape)
		cv2.imwrite(folder_name + str(i) + ".png", image)
		
		if verbose:
			cv2.imshow("image", image)
			cv2.waitKey()
		
		if i % 100 == 0:
			print(i)


# revise the latent class specification, depending on the 
# given labels; we know what labels map to what classes
# across the different factors of variation
def revise_latent_spec(latent_spec, label, mappings):
	
	mappings_keys = mappings.keys()

	for key in label:
		for mkey in mappings_keys:
			if key in mappings[mkey].keys():
				new_value = mappings[mkey][key]
				latent_spec[mkey] = [new_value]
				break
		
	return latent_spec


if __name__ == "__main__":
	
	args = parser.parse_args()

	# Given Latent Classes
	# [0] Color: white
	# [1] Shape: square, ellipse, heart
	# [2] Scale: 6 values linearly spaced in [0.5, 1]
	# [3] Orientation: 40 values in [0, 2 pi]
	# [4] Position X: 32 values in [0, 1]
	# [5] Position Y: 32 values in [0, 1]

	# in order to be able to refine out latent specs wrt to
	# user-defined labels we need to know what do these
	# labels mean wrt to the latent factors
	mappings = {}
	mappings['color'] = {'white': numpy.array([255, 255, 255]),
						 'red': numpy.array([0, 0, 255]),
						 'yellow': numpy.array([0, 255, 255]),
						 'green': numpy.array([0, 255, 0]),
						 'blue': numpy.array([255, 0, 0]),
						 'pink': numpy.array([255, 0, 255])
						 }

	mappings['shape'] = {'square': 0,
				   		 'ellipse': 1,
				   		 'heart': 2}

	mappings['scale'] = {'small': 0,
				   		 'big': 5}

	# describes the specification wrt to which we filter the 
	# images, depending on their latent factor classes
	# the spec is refined once we are given labels
	latent_spec = {'shape':range(3),
				   'scale':range(6),
				   'orientation':range(40),
				   'x':range(32),
				   'y':range(32)}

	# the labels we are interested in
	label_groups = {'shape':['square', 'ellipse'],
			   		'scale':['small', 'big'],
			   		'colors': ['white', 'pink']}

	# delete any previous object folders
	folder_name = "data/dSprites/"
	prep_dir(folder_name)

	# build up the labels for all objects - take the combinations of the
	# lists in label_groups; color is a special case, since we add it - it is
	# not part of the given latent factors of variation
	labels_but_color = list(itertools.product(*[label_groups[x] for x in label_groups if x != 'colors']))
	colors = label_groups['colors']
	
	# extract images for each possible label combination from the given groups and 
	# export in the relevant folders
	for label in labels_but_color:
		for color in colors:
			object_folder_name = folder_name + '_'.join(label) + '_' + color + "/"
			os.makedirs(object_folder_name)

			revised_latent_spec = revise_latent_spec(copy.deepcopy(latent_spec), label, mappings)
			extract(folder_name=object_folder_name, latent_spec=revised_latent_spec, image_size=args.image_size, cutoff=args.cutoff, bgr_color=mappings['color'][color])
