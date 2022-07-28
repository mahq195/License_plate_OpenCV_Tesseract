# import the necessary packages
import numpy as np
import cv2
import os
import imutils
from four_points_transform import four_points_transform

def bb_intersection_over_union(points, line):

	boxA = transform_predicted_box(points)
	# print(boxA)
	boxB = transform_ground_truth(line)
	# print(boxB)

	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2] + boxA[0], boxB[2] + boxB[0])
	yB = min(boxA[3] + boxA[1], boxB[3] + boxB[1])
	# compute the area of intersection rectangle
	interArea = abs(xB - xA + 1) * abs( yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = boxA[2] * boxA[3] 
	boxBArea = boxB[2] * boxB[3] 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def transform_predicted_box(points):
	x = min(points[0][0], points[3][0])
	y = min(points[0][1], points[1][1])
	w = max(points[1][0], points[2][0]) -x
	h = max(points[2][1], points[3][1]) -y
	boxA = [x, y, w, h]
	return list(map(int,boxA))

def transform_ground_truth(line): # example "0000_08244_b.jpg 1 181 75 79 70"
	lis = line.split()
	boxB = lis[2:] 
	return list(map(int,boxB))

#################
def get_coordinates(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    blured = cv2.bilateralFilter(gray, 11, 17, 17)
    ret,th1 = cv2.threshold(blured,150,255,cv2.THRESH_BINARY)
    edged = cv2.Canny(th1, 100, 200)

    dilated = cv2.dilate(edged, (3,3))

    keypoints = cv2.findContours(dilated.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
    cnts = imutils.grab_contours(keypoints)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]


    location = None
    # cropped_image = None
    # points = None
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        # print(peri)
        approx = cv2.approxPolyDP(cnt, 0.05*peri , True)
        if len(approx) == 4 and cv2.contourArea(cnt) > 3000:
            location = approx
            break

    if location is not None:
        v1 = location[0][0]
        v2 = location[1][0]
        v3 = location[2][0]
        v4 = location[3][0]
        points =  np.array([v1, v2, v3, v4])
        cropped_image, order_points = four_points_transform(gray, points) # get bird-eye view
        # cv2.imshow('croped-image', cv2.resize(cropped_image, (0,0), fx=3, fy=3))
        global ord_points 
        ord_points = [list(map(int, point)) for point in order_points]
    return ord_points


with open("locations.txt", "r") as gr_truth:
    gts = gr_truth.readlines()

DATA_PATH = r'test_images'
list_images = os.listdir(DATA_PATH)
iou_metric = []

for image in list_images:
	image_path = os.path.join(DATA_PATH, image)
	img = cv2.imread(image_path)
	points = get_coordinates(img)
	iou = 0
	if points is not None:
		for gt in gts:
			if gt.startswith(image):
				iou = bb_intersection_over_union(points, gt)
	iou_metric.append(iou)

print('IOU:',sum(iou_metric)/len(iou_metric))

