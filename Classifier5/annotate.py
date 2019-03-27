# python annotate.py --input data/Beer --annot data_labeled

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True)
ap.add_argument("-a", "--annot", required=True)
args = vars(ap.parse_args())

try:
    if not os.path.exists(args["annot"]):
        os.makedirs("annot")
except OSError:
    print ("[Error] Can't create annotation directory")

imagePaths = list(paths.list_images(args["input"]))
counts = {}

for (i, imagePath) in enumerate(imagePaths):
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))

	try:
		image = cv2.imread(imagePath)
		cv2.imshow("Image", imutils.resize(image, width=1000))
		key = cv2.waitKey(0)

		if key == ord("."):
			print("[INFO] ignoring character")
			continue

		key = chr(key).upper()

		if key == '1':
			dirPath = os.path.sep.join([args["annot"], "water"])
		elif key == '2':
			dirPath = os.path.sep.join([args["annot"], "beer"])
		elif key == '3':
			dirPath = os.path.sep.join([args["annot"], "softdrink"])
		elif key == '4':
			dirPath = os.path.sep.join([args["annot"], "wine"])
		elif key == '5':
			dirPath = os.path.sep.join([args["annot"], "schnaps"])
		else:
			print("[INFO] invalid character, skipping image...")

		try:
			if not os.path.exists(dirPath):
				os.makedirs(dirPath)
		except OSError:
			print("[INFO] error creating directory")

		count = counts.get(key, 1)
		p = os.path.sep.join([dirPath, "{}.png".format(str(count).zfill(6))])
		cv2.imwrite(p, image)
		counts[key] = count + 1

	except KeyboardInterrupt:
		print("[INFO] manually leaving script")
		break

	except:
		print("[INFO] an error occured, skipping image...")