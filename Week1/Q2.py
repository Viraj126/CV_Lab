# Write a simple program to read and display a video file.
import cv2

cap = cv2.VideoCapture('moving_cat.mp4')

while cap.isOpened():
    if not cap.isOpened():
        print("Error establishing connection")

    ret, frame = cap.read()
    if ret:
        cv2.imshow('Displaying image frames from a webcam', frame)

    if cv2.waitKey(25) == 27:
        break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()
