import os
import cv2
import numpy as np
from PIL import Image

recognizer=cv2.createLBPHFaceRecognizer();
path='dataSet'

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #create empth face list
    print imagePaths
    faceSamples=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L');
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faceSamples.append(imageNp)
        Ids.append(Id)
        print Id

        cv2.imshow("training",imageNp)
        cv2.waitKey(10)
    return Ids, faceSamples

IDs,faceSamples=getImagesAndLabels(path)
recognizer.train(faceSamples,np.array(IDs))
recognizer.save('trainner/trainner.yml')
cv2.destroyAllWindows()