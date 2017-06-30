import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.createLBPHFaceRecognizer()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def getImages(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #create empth face list
    sampleFaces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images from the dataSets
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the stored image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the image_loader sample
        faces=detector.detectMultiScale(imageNp)
        #If a face is there then append that in the list as well as Id of it
        for (x,y,w,h) in faces:
            sampleFaces.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return sampleFaces,Ids

faces,Ids = getImages('datasets')
recognizer.train(faces, np.array(Ids))
recognizer.save('imageloader/imageloader.yml')