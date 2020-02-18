# Usage
# python MiniVGGNet_TL_Train.py --training dataset/training/ --test dataset/test/ --weights output/weights/tl_weights.hdf5 --report output/report/ --plot output/plot/MiniVGGNet_TL_Plot



# Import needed packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from deepLearningLib.preprocessing import ImageToArrayPreprocessor
from deepLearningLib.preprocessing import AspectAwarePreprocessor
from deepLearningLib.datasets import SimpleDatasetLoader
from deepLearningLib.nn.conv import MiniVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os



# Construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True, help="Path to input training dataset")
ap.add_argument("-T", "--testing", required=True, help="Path to input testing dataset")
ap.add_argument("-w", "--weights", required=True, help="Path to best model weights output file")
ap.add_argument("-r", "--report", required=True, help="Path to report output directory")
ap.add_argument("-p", "--plot", required=True, help="Path to output accuracy/loss plot")
args = vars(ap.parse_args())



# Define learning rate decay function
def step_decay(epoch):
	# Init base learning rate, drop factor, and epochs to drop
	initAlpha = 0.05
	factor = 0.5
	dropEvery = 5
	
	# Compute learning rate for current epoch
	alpha = initAlpha * (factor**np.floor((1+epoch)/dropEvery))
	
	return float(alpha)



# Grab list of images, extract class label names from image paths
print("[INFO] Loading Images...")
trainImagePaths = list(paths.list_images(args["training"]))
testImagePaths =list(paths.list_images(args["testing"]))
classLabels = [pt.split(os.path.sep)[-2] for pt in trainImagePaths]
classLabels = [str(x) for x in np.unique(classLabels)]


# Init image preprocessors
aap = AspectAwarePreprocessor(64,64)
iap = ImageToArrayPreprocessor()

# Load dataset, scale px intensities [0,1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(trainX, trainY) = sdl.load(trainImagePaths, verbose=500)
(testX, testY) = sdl.load(testImagePaths, verbose=100)
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

# Label encoder
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# Constuct image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
						 horizontal_flip=True, fill_mode="nearest")



# Init the optimizer and model
print("[INFO] Compiling Model...")
opt = SGD(lr=0.05, momentum=0.5, nesterov=True)
model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(classLabels))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Construct callback to save best model (only), based on validation loss
checkpoint = ModelCheckpoint(args["weights"], monitor="val_loss", save_best_only=True, verbose=1)
callbacks = [checkpoint, LearningRateScheduler(step_decay)]

# Train network
print("[INFO] Training Network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY),
						steps_per_epoch=len(trainX)//32, epochs=100, callbacks=callbacks, verbose=1)

# Evaluate network
print("[INFO] Evaluating Network, Saving Report...")
predictions = model.predict(testX, batch_size=32)
report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classLabels)
			

	
# Save classification report
p = [args["report"], "report.txt"]
f = open(os.path.sep.join(p), "w")
f.write(report)
f.close()

				

# Plot training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="Training Loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="Testing Loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="Training Accuracy")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="Testing Accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])