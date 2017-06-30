# Import OpenCV2 for image processing
import cv2
# Start capturing video 
cam = cv2.VideoCapture(0)
# Detect object in video stream using Haarcascade Frontal Face
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Unique id for each person
face_id=raw_input('Please Enter your ID [eg:-1,2,3...]: \n')
# Initialize sample face image
sampleNum=0
# Start looping
while(True):
# Capture video frame
    ret, img = cam.read()
# Convert frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect frames of different sizes, list of faces rectangles
    faces = detector.detectMultiScale(gray, 1.3,5)
# Loops for each faces
    for (x,y,w,h) in faces:
# Crop the image frame into rectangle
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("dataSet/User."+face_id+'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
# Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('Dataset_Creator',img)
    #wait for 100 miliseconds thrn stop taking video.
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 50
    elif sampleNum==50:
        break
cam.release()
cv2.destroyAllWindows()
