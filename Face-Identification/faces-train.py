import cv2
import os
import numpy as np
from PIL import Image
import pickle
from datetime import datetime


def saveimage(path, frame):
  now = datetime.now()
  current_time = now.strftime("%H:%M:%S")
  img_item = current_time + ".png"
  cv2.imwrite(os.path.join(path , img_item), frame)


def main(ARGS):
 face_cascade_front = cv2.CascadeClassifier(os.path.join(os.getcwd(),"Face-Identification", 'cascades/data/haarcascade_frontalface_alt2.xml'))
 face_cascade_profile = cv2.CascadeClassifier(os.path.join(os.getcwd(),"Face-Identification",'cascades/data/haarcascade_profileface.xml'))

 recognizer = cv2.face.LBPHFaceRecognizer_create()

 BASE_DIR = os.path.join(os.getcwd(),"Data")
 image_dir = os.path.join(BASE_DIR, str(ARGS.name), "Pictures")

 cap = cv2.VideoCapture(os.path.join(str(ARGS.device))) #rtsp://192.168.2.190/ipcam.sdp

 #Used for sense of progression and to make sure not too much photos would be saved in one folder
 numofpics=0
 numofmaximumpics=80

 while(numofpics<numofmaximumpics):

     #Capture frame-by-frame
     ret, frame = cap.read()
     gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

     #Detect frontal faces
     faces = face_cascade_front.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
     for (x, y, w, h) in faces:
         saveimage(image_dir,frame)
         numofpics+=1
         print numofpics

     #Detect profile faces
     faces = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
     for (x, y, w, h) in faces:
         saveimage(image_dir,frame)
         numofpics+=1
         print numofpics

     # Display the resulting frame
     cv2.imshow('move in front of the camera or press q when ready',frame)
     if cv2.waitKey(20) & 0xFF == ord('q'):
         break

 #When everything done, release the capture
 cap.release()
 cv2.destroyAllWindows()

 current_id = 0
 label_ids = {}
 y_labels = []
 x_train = []

 for root, dirs, files in os.walk(BASE_DIR):
	 for file in files:
 		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
                        print root 
			label = os.path.basename(os.path.dirname(root)).replace(" ", "-").lower()
                        print label
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]
			
			pil_image = Image.open(path).convert("L") # Grayscale
			size = (550, 550)
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(final_image, "uint8")

			#Check for profile faces in pic
			faces = face_cascade_front.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)

			#Check for profile faces in pic
                        faces = face_cascade_profile.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)


 with open(os.path.join(os.getcwd(),"Face-Identification","pickles/face-labels.pickle"), 'wb') as f:
	 pickle.dump(label_ids, f)

 recognizer.train(x_train, np.array(y_labels))
 recognizer.save(os.path.join(os.getcwd(),"Face-Identification","recognizers/face-trainner.yml"))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train a visual profile for a user")

    parser.add_argument('-n', '--name', required=True,
                        help="Name of the new user")
    parser.add_argument('-d', '--device', default=0,
                        help="Specify the address of the camera you are filming from. e.g. 192.168.2.100:4747/video")
   
    ARGS = parser.parse_args()
    main(ARGS)

