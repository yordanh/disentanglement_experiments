import numpy as np
import cv2
import math
import h5py
import numpy

data = np.load("/home/yordan/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")

result = []

indecies = []
for i, c in enumerate(data['latents_classes']):
	if c[2] in [4,5] and c[4] in [15,16,17] and c[5] in [15,16,17]:
		indecies.append(i)

print(indecies)
print(len(indecies))

images = numpy.take(data['imgs'], indecies, axis=0)

print(len(images))
print(images.shape)

for i, image in enumerate(images):
	image = cv2.resize(image, (100, 100))
	cv2.imwrite("data/dSprites/" + str(i) + ".png", image * 255)
	
	if i % 100 == 0:
		print(image.shape)
		print(i)

cv2.destroyAllWindows()