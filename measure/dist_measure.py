import numpy as np
from imutils import resize, grab_contours
import cv2

low_red = np.array([0, 9, 157])
high_red = np.array([180, 104, 255])
low_yellow = np.array([17, 200, 166])
high_yellow = np.array([50, 245, 251])

# Define object specific variablesj
dist = 0
focal = 450
pixels = 30
width = 4


# jfind the distance from then camera
def get_dist(rectange_params, image):
    # find no of pixels covered
    pixels = rectange_params[1][0]
    print(pixels)
    # calculate distance
    dist = (width*focal)/pixels

    # Wrtie n the image
    image = cv2.putText(image, 'Distance from Camera in CM :', org,
                        font, 1, color, 2, cv2.LINE_AA)

    image = cv2.putText(image, str(dist), (110, 50), font,
                        fontScale, color, 1, cv2.LINE_AA)

    return image

# Extract Frames


cap = cv2.VideoCapture("./dist.mp4")


# basic constants for opencv Functs
kernel = np.ones((3, 3), 'uint8')
font = cv2.FONT_HERSHEY_SIMPLEX
org = (0, 20)
fontScale = 0.6
color = (0, 0, 255)
thickness = 2


cv2.namedWindow('Object Dist Measure ', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Dist Measure ', 700, 600)


# loop to capture video frames
while True:
    ret, img = cap.read()

    img = resize(img, width=600)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(img, (11, 11), 0)

    # predefined mask for green colour detection
    mask = cv2.inRange(hsv_img, low_yellow, high_yellow)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Remove Extra garbage from image
    # d_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)

    # find the histogram
    cont = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cont = grab_contours(cont)

    # cont = sorted(cont, key=cv2.contourArea, reverse=True)[:1]
    if cont and len(cont) > 0:

        rects = []

        for cnt in cont:

            # make sure it not too small
            if cv2.contourArea(cnt) > 1000:

                # draw rect of small part
                minEnclosingTri = cv2.boundingRect(cnt)
                rect = cv2.minAreaRect(cnt)

                # check if the rect big enough
                if minEnclosingTri[2] < 10 or minEnclosingTri[3] < 10:
                    continue

                # save the data so we can find max min of x y later
                [x, y, w, h] = minEnclosingTri
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                img = get_dist(rect, img)

#     for cnt in cont:
#         # check for contour area
#         if (cv2.contourArea(cnt) > 100 and cv2.contourArea(cnt) < 306000):
# 
#             # Draw a rectange on the contour
#             rect = cv2.minAreaRect(cnt)
#             box = cv2.boxPoints(rect)
#             box = np.int0(box)
#             cv2.drawContours(img, [box], -1, (255, 0, 0), 3)
# 
#             img = get_dist(rect, img)

    cv2.imshow('Object Dist Measure ', img)
    cv2.imshow('Mask ', hsv_img)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
