# Import OpenCV2 for image processing
# Import os for file pathimport cv2, os
# Import numpy for matrix calculation
# Import Python Image Library (PIL)
import cv2, os
import numpy as np
from PIL import Image
# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.createLBPHFaceRecognizer()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
# Create method to get the images and label data
def getImages(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #create empth face list
    faceSamples=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces=detector.detectMultiScale(imageNp)
        #If a face is there then append that in the list as well as Id of it
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
# Pass the face array and IDs array
    return faceSamples,Ids
# Get the faces and IDs
faces,Ids = getImages('dataSet')
# Load the model using the faces and IDs
recognizer.train(faces, np.array(Ids))
# Save the model into trainner.yml
recognizer.save('trainner/trainner.yml')
