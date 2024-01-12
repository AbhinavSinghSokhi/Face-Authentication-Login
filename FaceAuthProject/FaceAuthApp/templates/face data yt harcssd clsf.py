import cv2 as cv
import numpy as np

cap= cv.VideoCapture(0)
face_cascade= cv.CascadeClassifier("E:/_Coding/openCV/Projects/real_world_projects/Login-SignUp_Using_Face_Authentication/FaceAuthProject/FaceAuthApp/templates/haarcascade_frontalface_alt.xml")

skip=0
face_data=[]
dataset_path="E:/_Coding/openCV/Projects/real_world_projects/Login-SignUp_Using_Face_Authentication/FaceAuthProject/FaceAuthApp/static/face_dataset/"

file_name= input("Enter the name of person: ")

while True:
    ret,frame=cap.read()

    gray_frame=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if ret==False:
        continue
    faces=face_cascade.detectMultiScale(gray_frame, 1.3,5)
    if len(faces)==0:
        continue
    k=1
    faces= sorted (faces, key =lambda x:x[2]*x[3], reverse=True)
    skip+=1
    count=str(skip)
    frame=cv.putText(frame, count, (20,60), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    for face in faces[:1]:
        x,y,w,h= face
        offset=5
        face_offset = frame[y-offset: y+h+offset,x-offset:x+w+offset]
        face_selection= cv.resize(face_offset, (100,100))
        if skip % 10==0:
            face_data.append(face_selection)
            print(len(face_data))
        cv.imshow(str(k),face_selection)
        k+=1

        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv.imshow("Faces", frame)

    if cv.waitKey(1) and skip==300:
    # if count==30:
        break
face_data= np.array(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

np.save(dataset_path + file_name, face_data)
print("Dataset saved at : {}".format(dataset_path + file_name + '.npy'))

cap.release()
cv.destroyAllWindows()