# USAGE
# python drone.py --video FlightDemo.mp4

# import the necessary packages
from collections import deque
import argparse
import imutils
import cv2
import pdb
import numpy as np
import threading
import pyttsx3
from imutils.video import VideoStream
from operator import xor
import time


CONST_DETECTION = 'D:\Hackaroo\working\print\\bar2.png'
CONST_VIDEO = 'first_demo.mp4'
CONST_RANGE = 'D:\Hackaroo\working\print\\bar2.png'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())


###########################################################################
###########################################################################
def mapp(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew



#image = cv2.imread("test_img.jpg")  # read in the image

def first():
    camera = cv2.VideoCapture(0)
    while True:
        return_value,img = camera.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow('image',gray)
        if cv2.waitKey(1)& 0xFF == ord('s'):
            cv2.imwrite('testw.jpg',img)
            break
    camera.release()
    cv2.destroyAllWindows()
    image=img



    image = cv2.resize(image, (1300, 800))  # resizing because opencv does not work well with bigger images
    orig = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB To Gray Scale
    #cv2.imshow("Title", gray)

    blurred = cv2.GaussianBlur(gray, (5, 5),
                               0)  # (5,5) is the kernel size and 0 is sigma that determines the amount of blur
    #cv2.imshow("Blur", blurred)

    edged = cv2.Canny(blurred, 30, 50)  # 30 MinThreshold and 50 is the MaxThreshold
    #cv2.imshow("Canny", edged)

    image, contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST,
                                                  cv2.CHAIN_APPROX_SIMPLE)  # retrieve the contours as a list, with simple apprximation model
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # the loop extracts the boundary contours of the page
    for c in contours:
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * p, True)

        if len(approx) == 4:
            target = approx
            break
    approx = mapp(target)  # find endpoints of the sheet

    pts = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])  # map to 800*800 target window

    op = cv2.getPerspectiveTransform(approx, pts)  # get the top or bird eye view effect
    dst = cv2.warpPerspective(orig, op, (800, 800))

    cv2.imshow("Scanned", dst)

    ########################################################################################


    img=dst

    return img
#img = cv2.imread("bar.png ", cv2.IMREAD_GRAYSCALE)  # queryiamge

###########################################################################
###########################################################################


#CONST_DETECTION=first()


count = 0

inches = 0
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
KNOWN_DISTANCE = 24.0

# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 11.0

#sift loading

img = cv2.imread(CONST_DETECTION)
sift = cv2.xfeatures2d.SIFT_create()
kp_image, desc_image = sift.detectAndCompute(img, None)
# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)





########################################################

def callback(value):
    pass


def setup_trackbars(range_filter):
    cv2.namedWindow("Trackbars", 0)

    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 255

        for j in range_filter:
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--filter', required=True,
                    help='Range filter. RGB or HSV')
    ap.add_argument('-i', '--image', required=False,
                    help='Path to the image')
    ap.add_argument('-w', '--webcam', required=False,
                    help='Use webcam', action='store_true')
    ap.add_argument('-p', '--preview', required=False,
                    help='Show a preview of the image after applying the mask',
                    action='store_true')
    args = vars(ap.parse_args())

    if not xor(bool(args['image']), bool(args['webcam'])):
        ap.error("Please specify only one image source")

    if not args['filter'].upper() in ['RGB', 'HSV']:
        ap.error("Please speciy a correct filter.")

    return args


def get_trackbar_values(range_filter):
    values = []

    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)

    return values



def main_range():
    # args = get_arguments()
    args = {}

    args['filter'] = 'HSV'
    args['image'] = CONST_RANGE
    args['webcam'] = False
    args['preview'] = False
    range_filter = args['filter'].upper()

    if args['image']:
        image = cv2.imread(args['image'])

        if range_filter == 'RGB':
            frame_to_thresh = image.copy()
        else:
            frame_to_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        camera = cv2.VideoCapture(0)

    setup_trackbars(range_filter)

    while True:
        if args['webcam']:
            ret, image = camera.read()

            if not ret:
                break

            if range_filter == 'RGB':
                frame_to_thresh = image.copy()
            else:
                frame_to_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values(range_filter)

        thresh = cv2.inRange(frame_to_thresh, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

        if args['preview']:
            preview = cv2.bitwise_and(image, image, mask=thresh)
            cv2.imshow("Preview", preview)
        else:
            cv2.imshow("Original", image)
            cv2.imshow("Thresh", thresh)

        if cv2.waitKey(1) & 0xFF is ord('q'):
            cv2.destroyAllWindows()
            break
        file = 'values.txt'
        main_value = str(v1_min)+','+str(v2_min)+','+str(v3_min)+','+str(v1_max)+','+str(v2_max)+','+str(v3_max)
        with open(file, 'w') as filetowrite:
            filetowrite.write(main_value)
            filetowrite.close()



######################################################



def fun(image):
	# pdb.set_trace()
	marker = find_marker(image)
	inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

	# draw a bounding box around the image and display it
	box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
	box = np.int0(box)
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
	# cv2.putText(image, "%.2fft" % (inches / 12),
	# 			(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
	# 			2.0, (0, 255, 0), 3)
	# print(inches / 12)
	return inches


def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)

	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key = cv2.contourArea)

	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth



# load the video


if args.get("video", False):
	vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(CONST_VIDEO)
count = 0

def speak(text):

	# pdb.set_trace()
	engine.say(text)
	engine.runAndWait()
	return

image = cv2.imread("print\\white.jpg")
marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

# keep looping

# greenLower = (91, 10, 137)
# greenUpper = (100, 255, 235)
def color_mask(frame,cons):

	f = open('values.txt','r')
	e = f.readline()
	f.close()
	e = e.split(',')
	e = [int(val) for val in e]
	greenLower = (e[0]-cons, e[1]-cons, e[2]-cons)
	greenUpper = (e[3], e[4], e[5])

	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	if len(cnts) > 0:
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		if radius > 10:

			#cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
			#cv2.circle(frame, center, 5, (0, 0, 255), -1)
			bluecnts = cv2.findContours(mask.copy(),
										cv2.RETR_EXTERNAL,
										cv2.CHAIN_APPROX_SIMPLE)[-2]

			if len(bluecnts) > 0:
				blue_area = max(bluecnts, key=cv2.contourArea)
				(xg, yg, wg, hg) = cv2.boundingRect(blue_area)
				#cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)
				cv2.rectangle(mask, (xg, yg), (xg + wg, yg + hg), 255, -1)
				frame = cv2.bitwise_and(frame,frame, mask=mask)
	return frame


def syf(frame):
	frame1 = frame
	for i in range(10):
		frame = color_mask(frame,i*0.8)
	# frame = color_mask(frame)
	grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage
	kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
	try:
		matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
	except:
		return (frame,True)
	good_points = []
	try:
		for m, n in matches:
			if m.distance < 0.6 * n.distance:
				good_points.append(m)
	except:
		return frame
	# img3 = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)
	# Homography
	if len(good_points) > 10:
		query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
		train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

		matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
		matches_mask = mask.ravel().tolist()
		# Perspective transform

		if matrix is not None:
			h, w = img.shape
			pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
			dst = cv2.perspectiveTransform(pts, matrix)
		else:
			dst = []
		# dst = cv2.perspectiveTransform(pts, matrix)
		# print('dist: ', dst)

		homography = cv2.polylines(frame1, [np.int32(dst)], True, (255, 0, 0), 3)
		# cv2.imshow("Homography", homography)
		return (homography,False)
	else:
		return (frame1,True)
		# cv2.imshow("Homography", grayframe)

def main():
	while True:
		# grab the current frame and initialize the status text
		(grabbed,frame) = vs.read()
		status = "No Targets"

		# check to see if we have reached the end of the
		# video
		if not grabbed:
			break
		flag = False
		# convert the frame to grayscale, blur it, and detect edges
		try:
			frame, flag = syf(frame)
		except:
			pass
		if flag == False:
			cv2.imshow("Frame", frame)
			# print(count+1)
			key = cv2.waitKey(1) & 0xFF

			# if the 'q' key is pressed, stop the loop
			if key == ord("q"):
				break

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (7, 7), 0)
		edged = cv2.Canny(blurred, 50, 150)

		# find contours in the edge map
		cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		# loop over the contours
		for c in cnts:
			# approximate the contour
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.01 * peri, True)

			# ensure that the approximated contour is "roughly" rectangular
			if len(approx) >= 4 and len(approx) <= 6:
				# compute the bounding box of the approximated contour and
				# use the bounding box to compute the aspect ratio
				(x, y, w, h) = cv2.boundingRect(approx)
				aspectRatio = w / float(h)

				# compute the solidity of the original contour
				area = cv2.contourArea(c)
				hullArea = cv2.contourArea(cv2.convexHull(c))
				solidity = area / float(hullArea)

				# compute whether or not the width and height, solidity, and
				# aspect ratio of the contour falls within appropriate bounds
				keepDims = w > 15 and h > 15
				keepSolidity = solidity > 0.9
				keepAspectRatio = aspectRatio >= 0.6 and aspectRatio <= 1.4

				# ensure that the contour passes all our tests
				if keepDims and keepSolidity and keepAspectRatio:
					# draw an outline around the target and update the status
					# text

					# if flag:
					cv2.drawContours(frame, [approx], -1, (0, 0, 255), 2 )

					status = "Target(s) Acquired"
					try:
						if status == "Target(s) Acquired" and count == 100:
							count = 0
							inchess = fun(frame)
							print("%.2fft" % (inchess / 12))
							text = ''
							if int(inches)<int(inchess):
								text = 'move closer'
							else:
								text = 'good'
							inches = inchess
							thr = threading.Thread(target=speak, args=([text]), kwargs={})
							thr.start()  # Will run "foo"


						else:
							count = count+1


					except:
						pass
					#

					# compute the center of the contour region and draw the
					# crosshairs
					M = cv2.moments(approx)
					(cX, cY) = (int(M["m10"] // M["m00"]), int(M["m01"] // M["m00"]))
					(startX, endX) = (int(cX - (w * 0.15)), int(cX + (w * 0.15)))
					(startY, endY) = (int(cY - (h * 0.15)), int(cY + (h * 0.15)))
					cv2.line(frame, (startX, cY), (endX, cY), (0, 0, 255), 3)
					cv2.line(frame, (cX, startY), (cX, endY), (0, 0, 255), 3)

		# draw the status text on the frame
		cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
			(0, 0, 255), 2)

		# show the frame and record if a key is pressed
		cv2.imshow("Frame", frame)
		# print(count+1)
		key = cv2.waitKey(1) & 0xFF

		# if the 'q' key is pressed, stop the loop
		if key == ord("q"):
			break
		elif key == ord("r"):
			main_range()
	# vs.release()
	cv2.destroyAllWindows()
# cleanup the camera and close any open windows


if __name__ == '__main__':
    main()