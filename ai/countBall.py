import cv2
import numpy as np
from imutils import resize, grab_contours
from collections import deque
from time import perf_counter


class ballTracker:
    # find contour of balls, return frame and mask
    buffer = 64
    pts = deque(maxlen=buffer)

    def track(self, frame, upper, lower):
        frame = resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # turn gray
        # gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # find contours in the mask and initialize the current
        # (x, y) center of the ball

        cnts = cv2.findContours(mask.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = grab_contours(cnts)

        center = None
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)),
                           int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
        # update the points queue
        self.pts.appendleft(center)
        # loop over the set of tracked points
        for i in range(1, len(self.pts)):
            # if either of the tracked points are None, ignore
            # them
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue
            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            # thickness = int(np.sqrt(buffer / float(i + 1)) * 2.5)
            # cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
            frame = cv2.bitwise_and(frame, frame, mask=mask)
        return frame, mask


class ballCounter:
    # call the number of balls
    def __init__(self) -> None:
        pass

    def countBall(self, frame, cnts):
        # print(cnts)
        ball = 0
        x = 0
        y = 0
        x2 = 0
        y2 = 0
        returnFrame = frame
        # check if there was something
        if cnts and len(cnts) > 0:

            rects = []

            for cnt in cnts:

                # make sure it not too small
                if cv2.contourArea(cnt) > 2000:

                    # draw rect of small part
                    minEnclosingTri = cv2.boundingRect(cnt)

                    # check if the rect big enough
                    if minEnclosingTri[2] < 10 or minEnclosingTri[3] < 10:
                        continue

                    # save the data so we can find max min of x y later
                    [x, y, w, h] = minEnclosingTri
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    rects.append(minEnclosingTri)

            # sort the data
            x_values = self.column(rects, 0)
            y_values = self.column(rects, 1)
            widths = self.column(rects, 2)
            heights = self.column(rects, 3)

            # find max min

            if len(x_values) > 0:
                x = min(x_values)
            if len(y_values) > 0:
                y = min(y_values)
            if len(widths) > 0:
                x2 = max(self.plus(x_values, widths))
            if len(heights) > 0:
                y2 = max(self.plus(y_values, heights))

            # draw bing rect, this will represent the whole ball
            # or multiple balls
            # testframe = cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
            # cv2.imshow("test", testframe)

            # 1 ball 1:1, 2 balls 1:2, 3 balls 1:3
            rectRatio = 0
            if x - x2 != 0:
                rectRatio = (y - y2) / (x - x2)
            ball += round(rectRatio)
        return [x, y, x2, y2], ball

    def column(self, matrix, i):
        return [row[i] for row in matrix]

    def plus(self, list1, list2):
        returnList = []
        if len(list1) != len(list2):
            Exception
        for i in range(len(list1)):
            returnList.append(list1[i] + list2[i])
        return returnList


class countBallInSilo:

    def __init__(self) -> None:
        pass

    ballCounter = ballCounter()
    tracker = ballTracker()

    # oldNumRed = 0
    # oldNumBlue = 0
    # oldNumYellow = 0

    # Yellow color
    # yellow 17 50 200 245 166 251
    low_yellow = np.array([17, 200, 166])
    high_yellow = np.array([50, 245, 251])

    # Blue color
    # blue 11 103 6 220 143 248
    # low_blue = np.array([11, 6, 143])
    # high_blue = np.array([103, 220, 248])
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([126, 255, 255])
    # Red color
    # red 6 46 210 255 150 255
    # low_red = np.array([0,80,120])
    # high_red = np.array([10,255,255])
    low_red = np.array([0, 9, 157])
    high_red = np.array([180, 104, 255])

    # call the number of balls

    def count(self, frame, hsv):

        red, red_mask = self.tracker.track(hsv.copy(), self.high_red, self.low_red)
        cnts = cv2.findContours(red_mask.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        # red_mask = cv2.inRange(hsv_frame, low_red, high_red)
        # red = cv2.bitwise_and(frame, frame, mask=red_mask)

        blue_mask = cv2.inRange(hsv, self.low_blue, self.high_blue)
        # blue = cv2.bitwise_and(frame, frame, mask=blue_mask)
        # blue, blue_mask = tracker.track(hsv_frame.copy(),high_blue, low_blue)
        # cnts = cv2.findContours(red_mask.copy(), cv3.RETR_EXTERNAL,
        #   cv2.CHAIN_APPROX_SIMPLE)

        yellow, yellow_mask = self.tracker.track(hsv.copy(),
                                                 self.high_yellow,
                                                 self.low_yellow)

        cnts = cv2.findContours(red_mask.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

        cnts = grab_contours(cnts)

        # center = None

        boundingBox = [None, None, None]
        boundingBox[0], self.numRedBall = self.ballCounter.countBall(frame.copy(), grab_contours(cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)))
        boundingBox[1], self.numBlueBall = self.ballCounter.countBall(frame.copy(), grab_contours(cv2.findContours(blue_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)))
        boundingBox[2], self.numYellowBall = self.ballCounter.countBall(frame.copy(), grab_contours(cv2.findContours(yellow_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)))
        
        for box in boundingBox:
            x = box[0]
            y = box[1]
            x2 = box[2]
            y2 = box[3]
            cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 255), 2)


        # if self.oldNumRed != self.numRedBall or self.oldNumBlue != self.numBlueBall or self.oldNumYellow != self.numYellowBall:
        #     self.oldNumRed = self.numRedBall
        #     self.oldNumBlue = self.numBlueBall
        #     self.oldNumYellow = self.numYellowBall
        #     return [[self.red_frame, self.blue_frame, self.yellow_frame],
        #             [self.numRedBall, self.numBlueBall, self.numYellowBall]]
        # else:
        #     return [self.red_frame, self.blue_frame, self.yellow_frame]

        return [self.numRedBall, self.numBlueBall, self.numYellowBall], frame


def show3Frame(frames):
    if len(frames) == 3:
        cv2.imshow("Red", frames[0])
        cv2.imshow("blue", frames[1])
        cv2.imshow("yellow", frames[2])


s = perf_counter()
# address = "http://192.168.31.240:8080/video"
# cap.open(address)


def main():
    counter = countBallInSilo()
    while True:
        cap = cv2.VideoCapture("./asset/3ball.jpeg")
        ret, frame = cap.read()

        if not ret:
            break
        frame = resize(frame, width=600)

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        result = counter.count(frame, hsv_frame)

        if len(result) == 2:
            cv2.imshow("the result", result[1])
            ball = result[0]
            print(f"red: {ball[0]} | blue: {ball[1]} | yellow: {ball[2]}")

        # print(f"executed in {elapsed:0.2f} seconds.")
        #    break

        key = cv2.waitKey(50)
        if key == ord('q'):
            break


