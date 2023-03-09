import cv2
import numpy as np
import face_recognition

imgGOODWIN = face_recognition.load_image_file("imagebasic/GOODWIN.jpeg")
imgGOODWIN = cv2.cvtColor(imgGOODWIN,cv2.COLOR_BGR2RGB)
imgASHWIN = face_recognition.load_image_file("imagebasic/Cristiano_Ronaldo.jpeg")
imgASHWIN = cv2.cvtColor(imgASHWIN,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgGOODWIN)[0]
encodeGOODWIN =face_recognition.face_encodings(imgGOODWIN)[0]
cv2.rectangle(imgGOODWIN,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
faceLocTEST= face_recognition.face_locations(imgASHWIN)[0]
encodeASHWIN =face_recognition.face_encodings(imgASHWIN)[0]
cv2.rectangle(imgASHWIN,(faceLocTEST[3],faceLocTEST[0]),(faceLocTEST[1],faceLocTEST[2]),(255,0,255),2)

result = face_recognition.compare_faces([encodeASHWIN],encodeGOODWIN)
facedistance = face_recognition.face_distance([encodeASHWIN],encodeGOODWIN)
print(result,facedistance)
cv2.putText(imgASHWIN,f'{result}{round(facedistance[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.imshow('GOODWiN',imgGOODWIN)
cv2.imshow('ANGEL',imgASHWIN)
cv2.waitKey(0)
