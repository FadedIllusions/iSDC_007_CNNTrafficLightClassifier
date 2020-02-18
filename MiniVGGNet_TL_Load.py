# Usage 
# python MiniVGGNet_TL_Load.py --images examples/ --model output/weights/tl_weights.hdf5



# Import needed packages
from deepLearningLib.preprocessing import ImageToArrayPreprocessor
from deepLearningLib.preprocessing import AspectAwarePreprocessor
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2



# Construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="Path to input images directory")
ap.add_argument("-m", "--model", required=True, help="Path to pre-trained model")
args = vars(ap.parse_args())



# Init labels
labelNames = ["green", "red", "yellow"]

# Load pre-trained network
print("[INFO] Loading Pre-Trained Network...")
model = load_model(args["model"])



# Load image paths
imagePaths = list(paths.list_images(args["images"]))

for i in range(0,len(imagePaths)):
	# Load and preprocess image
	print("[INFO] Loading Image {}...".format(i+1))
	image = cv2.imread(imagePaths[i])
	data = AspectAwarePreprocessor(64,64).preprocess(image)
	data = ImageToArrayPreprocessor().preprocess(data)
	data = data.astype("float")/255.0
	data = np.expand_dims(data, axis=0)

	# Make predictions on images
	print("[INFO] Classifying Image...")
	pred = model.predict(data).argmax(axis=1)	
	label = labelNames[int(pred)]

	# Display image and classification
	cv2.imshow(label, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
