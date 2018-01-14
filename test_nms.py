# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 21:48:28 2018

@author: Administrator
"""
# import the necessary packages
from nms import non_max_suppression_fast
import numpy as np
import cv2
 
# construct a list containing the images that will be examined
# along with their respective bounding boxes
images = [
	("images/audrey.jpg", np.array([
	(19, 40, 271, 299),
	(66, 92, 378, 393),
	(71, 62, 360, 309)])),
	("images/bksomels.jpg", np.array([
	(247, 183, 447, 470),
	(217, 260, 493, 553),
	(187, 139, 531, 831),
    (285, 308, 401, 585)])),
	("images/gpripe.jpg", np.array([
	(230, 117, 305, 193),
	(246, 134, 292, 218),
	(224, 105, 307, 253),
	(311, 122, 366, 209),
    (301, 122, 388, 191),
    (323, 105, 293, 236)]))]
 
# loop over the images
for (imagePath, boundingBoxes) in images:
	# load the image and clone it
	print ("[x] %d initial bounding boxes" % (len(boundingBoxes)))
	image = cv2.imread(imagePath)
	orig=np.zeros(image.shape,np.uint8)
	orig = image.copy()
 
	# loop over the bounding boxes for each image and draw them
	for (startX, startY, endX, endY) in boundingBoxes:
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)
 
	# perform non-maximum suppression on the bounding boxes
	pick = non_max_suppression_fast(boundingBoxes, 0.5)
	print ("[x] after applying non-maximum, %d bounding boxes" % (len(pick)))
 
	# loop over the picked bounding boxes and draw them
	for (startX, startY, endX, endY) in pick:
		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
 
	# display the images
	cv2.imshow("Original", orig)

	cv2.imshow("After NMS", image)
	cv2.waitKey(0)
































#import numpy as np
#import nms
#
#boxes=[[200,200,400,400,0.99],  
#        [220,220,420,420,0.9],  
#        [100,100,150,150,0.82],  
#        [200,240,400,440,0.5],
#        [150,250,300,400,0.88]] 
#boxes=np.array(boxes)
#
#abc=nms.non_max_suppression_slow(boxes,0.8)
##abc=nms.py_cpu_nms(boxes,0.8)
#print(abc)
