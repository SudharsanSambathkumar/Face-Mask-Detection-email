# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
from cv2 import *
import os
import smtplib
from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage



def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
print("[INFO] Get Ready!!!, Mask detection progrom going to start...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred



		# determine the class label and color we'll use to draw
		# the bounding box and text
		#if mask:
		#	label="Mask"
		#else:
		#	label="No Mask"
		label = "Mask" if mask>withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)







		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		#getting image of person who not wearing mask and sending through email
		if withoutMask > 0.5:
			img_name = "PATH TO SAVE THE  NOT WEARING MASK IMAGE"
			cv2.imwrite(img_name, frame)

			strFrom = 'SENDER@gmail.com'
			strTo = 'RECEIVER@gmail.com'#if the receiver is many use  array ['1@gmail.com,2@gmail.com']

			# Create the root message

			msgRoot = MIMEMultipart('related')
			msgRoot['Subject'] = 'subject of the email'#pic of not wearing mask
			msgRoot['From'] = strFrom
			msgRoot['To'] = strTo

			msgRoot.preamble = 'Multi-part message in MIME format.'
			msgAlternative = MIMEMultipart('alternative')
			msgRoot.attach(msgAlternative)

			msgText = MIMEText('Alternative plain text message.')
			msgAlternative.attach(msgText)

			#using HTML to read an image if u need edit the statements
			msgText = MIMEText('<b>important <i>HTML</i> text</b> and an image.<br><img src="cid:image1"></br>IMAGE-DATA!','html')			
			msgAlternative.attach(msgText)

			# Attach Image
			fp = open('PATH THAT SAVED THE  NOT WEARING MASK IMAGE', 'rb')  # Read the result image
			msgImage = MIMEImage(fp.read())
			fp.close()

			# Define the image's ID as referenced above
			msgImage.add_header('Content-ID', '<image1>')
			msgRoot.attach(msgImage)

			server = smtplib.SMTP('smtp.gmail.com', 587)
			server.starttls()
			server.login('SENDER@gmail.com', 'PASSWORD')  # Username and Password of Account
			server.sendmail(strFrom, strTo, msgRoot.as_string())
			server.quit()
			print("IF U NEED ANY MSG IN OUTPUT TERMINAL FOR PROGRAMMER USE")

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_ITALIC, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Output frame", frame)
	key = cv2.waitKey(1) & 0xFF
		












# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()