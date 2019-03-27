# python model_train_PAT.py --dataset Classifier5/data_labeled --model models/models_christopher/minivggnet_1

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import np_utils
from keras.optimizers import SGD
from imutils import paths
from cnn.nn.conv import LeNet
from cnn.nn.conv import ShallowNet
from cnn.nn.conv import MiniVGGNet
from cnn.preprocessing import ImageToArrayPreprocessor
from cnn.preprocessing import SimplePreprocessor
from cnn.datasets import SimpleDatasetLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
import talos as ta

batch_size = 64
epochs = 15
loss = "binary_crossentropy"
optimizer = SGD(lr=0.01, decay=0.01 / 15, momentum=0.9, nesterov=True) #adam, SGD(0.01), SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True

test_size = 0.20

input_width = 50
input_height = 50

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True)
ap.add_argument("-m", "--model", required=True)
args = vars(ap.parse_args())

imagePaths = paths.list_images(args["dataset"])

sp = SimplePreprocessor(input_width, input_height)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp,iap])
(data, labels) = sdl.load(imagePaths, verbose=500)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 5)

classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=test_size, stratify=labels, random_state=42)

print("[INFO] compiling model...")
model = MiniVGGNet.build(width=input_width, height=input_height, depth=3, classes=5)
model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	class_weight=classWeight, batch_size=batch_size, epochs=epochs, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=batch_size)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))

print("[INFO] serializing network...")
model.save(args["model"] + ".hdf5")

File = open(args["model"] + ".txt","w+")

File.write("batch size = " + str(batch_size) + "\n")
File.write("epochs = " + str(epochs) + "\n")
File.write("loss = " + str(loss) + "\n")
File.write("optimizer = " + str(optimizer) + "\n")
File.write("test size = " + str(test_size) + "\n\n")
File.write("train accuracy = \n")
File.write("train loss = \n")
File.write("test accuracy = \n")
File.write("test loss = \n\n")
File.write("average accuracy = \n\n")
File.write("input width = " + str(input_width) + "\n")
File.write("input height = " + str(input_height) + "\n")

File.close()

# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, 15), H.history["acc"], label="acc")
# plt.plot(np.arange(0, 15), H.history["val_acc"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend()
# plt.show()