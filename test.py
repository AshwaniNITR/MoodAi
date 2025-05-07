import cv2
print("OpenCV version:", cv2.__version__)
cv2.namedWindow("Test")
cv2.imshow("Test", cv2.imread("./Portfolio-potrait-3.jpg"))  # Use any valid image path
cv2.waitKey(0)
cv2.destroyAllWindows()
