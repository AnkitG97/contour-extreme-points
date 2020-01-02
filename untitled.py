import imutils
import cv2
from scipy.spatial import distance as dist

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread("button_01.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

cv2.namedWindow('Image',cv2.WINDOW_NORMAL )
cv2.imshow("Image", image)
cv2.waitKey(0)