# python augmentate.py --input dataset --output aug_dataset

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from imutils import paths
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True)
ap.add_argument("-o", "--output", required=True)
ap.add_argument("-p", "--prefix", type=str, default="augmented")
args = vars(ap.parse_args())

try:
    if not os.path.exists(args["output"]):
        os.makedirs("output")
except OSError:
    print ("[Error] Can't create output directory")

print("[INFO] loading example image...")
imagePaths = list(paths.list_images(args["input"]))

for imagePath in imagePaths:
	image = load_img(imagePath)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
		height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
		horizontal_flip=True, fill_mode="nearest")
	total = 0

	if not os.path.exists(args["output"] + "/" + imagePath.split(os.path.sep)[-2]):
			os.makedirs(args["output"] + "/" + imagePath.split(os.path.sep)[-2])

	print("[INFO] generating images...")
	imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"] + "/" + imagePath.split(os.path.sep)[-2],
		save_prefix=args["prefix"], save_format="jpg")

	for image in imageGen:
		total += 1

		if total == 10:
			break