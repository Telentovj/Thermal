# USAGE
# py main.py --video video.avi

# import the necessary packages
from imutils.video import FPS
import multiprocessing
import numpy as np
import argparse
import dlib
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from detect import detect


def start_tracker(box, label, rgb, inputQueue, outputQueue, refreshRate):
	frame = 1
	# construct a dlib rectangle object from the bounding box
	# coordinates and then start the correlation tracker
	t = dlib.correlation_tracker()
	rect = dlib.rectangle(box[0], box[1], box[2], box[3])
	t.start_track(rgb, rect)
	# loop indefinitely -- this function will be called as a daemon
	# process so we don't need to worry about joining it
	while True:
		# attempt to grab the next frame from the input queue
		rgb = inputQueue.get()
		frame += 1		
		# if there was an entry in our queue, process it
		if rgb is not None:
			# update the tracker and grab the position of the tracked
			# object
			buffer = t.update(rgb)
			pos = t.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())
			
			
			# add the label + bounding box coordinates to the output
			# queue
			if frame < refreshRate:
				outputQueue.put((label, (startX, startY, endX, endY)))
				# print(frame)
			else:
				outputQueue.put(('person', (0,0,0,0)))
				return
	
# helper function to convert bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        width = xmax - xmin
        height = ymax - ymin
        #box[0], box[1], box[2], box[3] = xmin, ymin, width, height     
        box[0] = xmin
        box[1] = ymin
        box[2] = xmax
        box[3] = ymax
    return bboxes

def yolov4(image_data,infer):
	batch_data = tf.constant(image_data)
	pred_bbox = infer(batch_data)
	print(pred_bbox)
	for key, value in pred_bbox.items():
		boxes = value[:, :, 0:4]
		pred_conf = value[:, :, 4:]

	boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
		boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
		scores=tf.reshape(
			pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
		max_output_size_per_class=50,
		max_total_size=50,
		iou_threshold=0.45,
		score_threshold=0.5
	)
	
	# convert data to numpy arrays and slice out unused elements
	num_objects = valid_detections.numpy()[0]
	bboxes = boxes.numpy()[0]
	bboxes = bboxes[0:int(num_objects)]
	scores = scores.numpy()[0]
	scores = scores[0:int(num_objects)]
	classes = classes.numpy()[0]
	classes = classes[0:int(num_objects)]
	original_h, original_w = 416,416
	bboxes = format_boxes(bboxes, original_h, original_w)

	return num_objects,bboxes,scores,classes

def fallAlgorithm(startX,startY,endX,endY):
	dx, dy = endX-startX, endY-startY  	# Check the difference
	difference = dy-dx	
	if difference < 0:						
		return (0,0,255)
	else:
		return 	(0,255,0)
    	
def startStream(args):
	# initialize our list of queues -- both input queue and output queue
	# for *every* object that we will be tracking
	inputQueues = []
	outputQueues = []

	# to track the frames
	totalFrames = 0

	# initialize the list of class labels MobileNet SSD was trained to
	# detect
	CLASSES = ["person"]

	# initialize the video stream and output video writer
	print("[INFO] starting video stream...")
	vs = cv2.VideoCapture(args["video"])
	writer = None

	# start the frames per second throughput estimator
	fps = FPS().start()
	# To monitor the multiprocessing.process
	processMonitor = []
	queueMonitor = {}
    # loop over frames from the video file stream
	while True:
		# grab the next frame from the video file
		(grabbed, frame) = vs.read()
		# check to see if we have reached the end of the video file
		if frame is None:
			break
		# resize the frame for faster processing and then convert the
		frame = cv2.resize(frame, (416,416))
		# print("Normal")
		# print(frame)
		# print("Gray")
		# print(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
		# frame from BGR to GRAY ordering 
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image_data = rgb
		image_data = image_data/ 255.
		image_data = image_data[np.newaxis, ...].astype(np.float32)
		# if we are supposed to be writing a video to disk, initialize
		# the writer
		if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(frame.shape[1], frame.shape[0]), True)

		
		# We refresh every 60 frames so that we can redetected the objects, then we track the detected objects
		# Hence we will be having 60 frame cycles
		if totalFrames%60 == 0:
    	
			# Ensure that the pervious process is terminated.
			# This line is not important if there are low detections
			# However, might be essential if there are >= 6 detections.
			for process in processMonitor:
				process.join()

			# Ensure input and output queues are empty so that we do not build up
			# excess queues when we redetect objects
			inputQueues.clear()
			outputQueues.clear()
			
			# Run detection algorithm
			(num_objects,bboxes,scores,classes) = detect(image_data,rgb)
			print("num_objects: " + str(num_objects))
			print("bboxes: ")
			print(bboxes)
			print(scores)
			print(classes)
			# loop over the detections
			for i in np.arange(0, num_objects):
				# extract the confidence (i.e., probability) associated
				# with the prediction
				
				confidence = scores

				# filter out weak detections by requiring a minimum
				# confidence
				if confidence[i] > args["confidence"]:
					# extract the index of the class label from the
					# detections list
					idx = int(classes[i].item())
					label = CLASSES[idx]

					# if the class label is not a person, ignore it
					if CLASSES[idx] != "person":
						continue

					# compute the (x, y)-coordinates of the bounding box
					# for the object
					box = bboxes[i]
					(startX, startY, endX, endY) = box.astype("int")
					bb = (startX, startY, endX, endY)
				

					# create two brand new input and output queues,
					# respectively
					iq = multiprocessing.Queue()
					oq = multiprocessing.Queue()
					refreshRate = 60 - i
					queueMonitor[oq] = iq
					inputQueues.append(iq)
					outputQueues.append(oq)
					# spawn a daemon process for a new object tracker
					p = multiprocessing.Process(
						target=start_tracker,
						args=(bb, label, rgb, iq, oq, refreshRate))
					processMonitor.append(p)
					p.daemon = True
					p.start()
					# grab the corresponding class label for the detection
					# and draw the bounding box
					# Detect falls here
					colour = fallAlgorithm(startX,startY,endX,endY)
					cv2.rectangle(frame, (startX, startY), (endX, endY),
						colour, 2)
					cv2.putText(frame, label+ " " + str(outputQueues.index(oq)), (startX, startY - 15),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2)

		# otherwise, we've already performed detection so let's track
		# multiple objects
		else:
			# loop over each of our input ques and add the input RGB
			# frame to it, enabling us to update each of the respective
			# object trackers running in separate processes
			for iq in inputQueues:
				iq.put(rgb)
				
				
			
			# loop over each of the output queues
			for oq in outputQueues:
				# grab the updated bounding box coordinates for the
				# object -- the .get method is a blocking operation so
				# this will pause our execution until the respective
				# process finishes the tracking update
				buffer2 = oq.get()

				# If we find that we lost the person or that we have reached the refresh frame
				# We proceed to terminate all the mulltiprocess queues 
				if buffer2 == ('person', (0,0,0,0)):
					removeIq = queueMonitor[oq]
					while not iq.empty(): 
						try:
							removeIq.get(timeout=0.005)
						except:
							pass
					# Closes the queue pipeline
					removeIq.close()
					# Flushes the queue
					removeIq.join_thread()
					# Removes the queue so that we don't have unneccesary queue
					inputQueues.remove(removeIq)
					
					while not oq.empty(): 
						try:
							oq.get(timeout=0.005)
						except:
							pass
					oq.close()
					oq.join_thread()				
					outputQueues.remove(oq)
				else:
					(label, (startX, startY, endX, endY)) = buffer2
					# draw the bounding box from the correlation object
					# Detect falls here
					colour = fallAlgorithm(startX,startY,endX,endY)
					cv2.rectangle(frame, (startX, startY), (endX, endY),
						colour, 2)
					cv2.putText(frame, label+ " " + str(outputQueues.index(oq)), (startX, startY - 15),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2)
		# check to see if we should write the frame to disk
	
		if writer is not None:
			writer.write(frame)
		
		# show the output frame
		cv2.imshow("Frame", frame)
		totalFrames += 1
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		# update the FPS counter
		fps.update()

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# check to see if we need to release the video writer pointer
	if writer is not None:
		writer.release()

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.release()
			


def driver():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video", default = 0,
		help="path to input video file")
	ap.add_argument("-o", "--output", type=str,
		help="path to optional output video file")
	ap.add_argument("-c", "--confidence", type=float, default=0.2,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())
	# Start streaming
	startStream(args)


if __name__ == '__main__':
	driver()