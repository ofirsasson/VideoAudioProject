import numpy as np
import cv2
import pickle
import cPickle
from datetime import datetime
from scipy.io.wavfile import read
from Classes.featureextraction import extract_features
import os
from Classes import audioclass2

def main(ARGS):

 def SpeechIdentifier(PathToTmpWav):

    for wavfile in os.listdir(PathToTmpWav):
 	if wavfile.endswith(".wav"):
         sr,audio = read(os.path.join(PathToTmpWav,wavfile))
    	 vector   = extract_features(audio,sr)
    	 log_likelihood = np.zeros(len(models)) 
    
    	 for i in range(len(models)):
        	gmm = models[i]  #Checking with each model one by one
        	scores = np.array(gmm.score(vector))
        	log_likelihood[i] = scores.sum()

    	 winner = np.argmax(log_likelihood)
    	 print speakers[winner]+": "+ os.path.splitext(wavfile)[0]
         os.remove(os.path.join(PathToTmpWav,wavfile))
         return speakers[winner]+": "+ os.path.splitext(wavfile)[0] 
    return " "

 def TextToScreen(oldframe):

    #Setting up parameters
    width = int(cap.get(3))
    height = int(cap.get(4))
    x, y, w, h = width/10, 8*height/10, 8 * width/10, height/10  # Rectangle parameters

    #Drawing a semi-transparent box for the text
    image = oldframe.copy()
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), -1)  # A filled rectangle
    alpha = 0.4  # Transparency factor.
    newframe = cv2.addWeighted(oldframe, alpha, image, 1 - alpha, 0)

    #Writing text on top of the box
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 0)
    stroke = 2
    cv2.putText(newframe, CurrentWordsSpoken, (x,int(y+h*0.5)), font, 1, color, stroke, cv2.LINE_AA)
    return newframe

 def FaceRecognizer(frame, gray, PathToRecognizer):

    #Loading cascade and finding roi(region of interest)
    face_cascade = cv2.CascadeClassifier(PathToRecognizer)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
    	roi_gray = gray[y:y+h, x:x+w]
    	roi_color = frame[y:y+h, x:x+w]

    	# Recognize
    	id_, conf = recognizer.predict(roi_gray)

        #Analyze results
    	if conf>=85:
    		font = cv2.FONT_HERSHEY_SIMPLEX
    		name = labels[id_]
    		color = (255, 255, 255)
    		stroke = 2
    		cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

                #Adding or renewing life points for speakers
                NewSpeaker = True
                for index, speaker in enumerate(CurrentSpeakers):
                    if name in speaker:
                        NewSpeaker = False
                        tmptuple = list(speaker)
                        tmptuple[1] = LifePoints
                        CurrentSpeakers[index]=tuple(tmptuple)
                if NewSpeaker:
                    CurrentSpeakers.append((name,LifePoints))
		print name + " detected on camera."

        color = (255, 0, 0) #BGR 0-255 
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)


 #Paths to folders
 FIPath = os.path.join(os.getcwd(),"Face-Identification")
 SRPath = os.path.join(os.getcwd(),"Speech-Recognition")
 SIPath = os.path.join(os.getcwd(),"Speech-Identification")
 DTPath = os.path.join(os.getcwd(),"Data")

 #Setting up recognizer
 recognizer = cv2.face.LBPHFaceRecognizer_create()
 recognizer.read("./Face-Identification/recognizers/face-trainner.yml")
 labels = {"person_name": 1}
 with open(os.path.join(FIPath ,"pickles/face-labels.pickle"), 'rb') as f:
 	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

 #Path where training speakers will be saved
 gmm_files = []
 for root, dirs, files in os.walk(DTPath):
    for tmpmodel in files:
        if tmpmodel.endswith('.gmm'):
            gmm_files.append(os.path.join(root,tmpmodel))

 #Load the Gaussian gender Models
 models = [cPickle.load(open(name,'r')) for name in gmm_files]
 speakers = [name.split("/")[-1].split(".gmm")[0] for name in gmm_files]

 #Setting up variables
 CurrentWordsSpoken=" "
 CurrentSpeakers = [] 
 LifePoints = 60



 #Setting up camera
 cap = cv2.VideoCapture(ARGS.device) #'rtsp://192.168.2.190/ipcam.sdp'


 while(True):

 ####Face-Recognition####
    #Capture frame-by-frame
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    FaceRecognizer(frame,gray,os.path.join(FIPath, "cascades/data/haarcascade_frontalface_alt2.xml"))
    FaceRecognizer(frame,gray,os.path.join(FIPath, "cascades/data/haarcascade_profileface.xml"))
    

#Updating life for speakers in the current frame 
    newspeakers = []
    for cleaner in CurrentSpeakers:
      if not cleaner[1] == 0:
        y = list(cleaner)
        y[1] -= 1
        newspeakers.append(tuple(y))
    CurrentSpeakers = newspeakers
    
 ####Speaker-Identification####
    word = SpeechIdentifier(os.path.join(DTPath ,"tmp"))
    
    for name in CurrentSpeakers:
        if not word == CurrentWordsSpoken and name[0] == word.split(":")[0]:
           CurrentWordsSpoken = word

    frame=TextToScreen(frame)

    #Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

 #When everything done, release the capture
 cap.release()
 cv2.destroyAllWindows()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Connecting Video and Audio together")
    parser.add_argument('-d', '--device', default=0,
                        help="Specify the address of the camera you are filming from. e.g. 192.168.2.100:4747/video")
  
    ARGS = parser.parse_args()
    main(ARGS)
