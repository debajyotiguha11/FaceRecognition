# Import OpenCV2 for image processing
import cv2
# Import numpy for matrices calculations
import numpy as np
# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.createLBPHFaceRecognizer()
# Load the image_trainner module
recognizer.load('trainner/trainner.yml')
# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"
# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Initialize and start the video frame capture
cam = cv2.VideoCapture(0)
# Set the font style
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
# Loop
while True:
# Read the video frame
    ret, im =cam.read()
# Convert the captured frame into grayscale
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
 # Get all face from the video frame
    faces=faceCascade.detectMultiScale(gray, 1.3,5)
# For each face in faces
    for(x,y,w,h) in faces:
# Create rectangle around the face
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
# Recognize the face belongs to which ID
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
# Check the ID if exist
        if(conf<50):
            if(Id==1):
                Id="Yourname"
            elif(Id==2):
                Id="Friend"
 #If not exist, then it is Unknown
        else:
            Id="Unknown"
# Put text describe who is in the picture
        cv2.cv.PutText(cv2.cv.fromarray(im),str(Id), (x,y+h),font, (0,255,0))
# Display the video frame with the bounded rectangle
    cv2.imshow('Face_Recognizer',im) 
# If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
# Stop the camera
# Close all windows
cam.release()
cv2.destroyAllWindows()
