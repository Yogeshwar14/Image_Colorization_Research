import numpy as np
import argparse
import cv2
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input black and white image")
ap.add_argument("-p", "--prototxt", type=str, required=True,
	help="path to Caffe prototxt file")
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--points", type=str, required=True,
	help="path to cluster center points")
args = vars(ap.parse_args())


print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
pts = np.load(args["points"])
# add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]


image = cv2.imread(args["image"])
scaled = image.astype("float32") / 255.0		#intensity [0-1]
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)	#bgr-lab	

resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]				#getting Luminousity
L -= 50							#normalizing L


'print("[INFO] colorizing image...")'
net.setInput(cv2.dnn.blobFromImage(L))				#passing L through network
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))		#predicting ab
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))		#original shape


L = cv2.split(lab)[0]								#getting original Luminousity
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)	#concentrating ab model according to L

colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)					#clipping [0-1] (>1 = 1)

colorized = (255 * colorized).astype("uint8")				#rescaling to 255

cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)

