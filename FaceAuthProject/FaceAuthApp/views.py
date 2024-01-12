import cv2 as cv
import os
import numpy as np
from django.shortcuts import render, redirect
from django.http import HttpResponse,JsonResponse,StreamingHttpResponse, HttpResponseServerError
from .models import User
from .forms import signupForm
from django.contrib import messages

# Create your views here.
def index(request):
    return render(request, "index.html")
    # return HttpResponse("Hello")

def login(request):
    return render(request, "loginpage.html")

def signup(request):

    if request.method=="POST":
        username = request.POST['username'] #values in brackets are name attribute of the input field
        email = request.POST['email']

        # face_data= capture_face_data(username)
        if len(username)<2:
            messages.error(request, "Fill the form correctly.")
        else:
            #saving code
            file_name= username
            user= User.objects.create(username=username, email=email)
            # user.face_data_file.save(username +'.npy',face_data)

            return redirect('loginpage')
    else:
        return render(request, "signupPage.html")




def generate(username):
    try:
        video_capture = cv.VideoCapture(0, cv.CAP_DSHOW)
        # cap= cv.VideoCapture(0)
        face_cascade= cv.CascadeClassifier("E:/_Coding/openCV/Projects/real_world_projects/Login-SignUp_Using_Face_Authentication/opencvtrystream/opencvstreamapp/static/haarcascade_frontalface_alt.xml")

        skip=0
        face_data=[]
        accumulated_face_data = []
        dataset_path="E:/_Coding/openCV/Projects/real_world_projects/Login-SignUp_Using_Face_Authentication/opencvtrystream/opencvstreamapp/static/face_datasets/"

        # file_name= input("Enter the name of person: ")

        file_name = username
        if not video_capture.isOpened():
            raise RuntimeError('Could not open video source.')
        print("Generating frames...")

        while True:
            success, frame = video_capture.read()
            gray_frame=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            if not success:
                print("Failed to capture frame.")
                continue

            #processing on video to take place here
            faces=face_cascade.detectMultiScale(gray_frame, 1.3,5)
            print(faces)
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
                # cv.imshow(str(k),face_selection)
                k+=1

                cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            # cv.imshow("Faces", frame) #because here the displaying of stream is happening using imencode

            if cv.waitKey(1) and skip==300:
            # if count==30:
                 break

            if len(face_data) > 0:
                accumulated_face_data.extend(face_data)

            if len(accumulated_face_data) > 0 :#and skip % 300 == 0:
                accumulated_face_data = np.array(accumulated_face_data)
                accumulated_face_data = accumulated_face_data.reshape((accumulated_face_data.shape[0], -1))
            np.save(dataset_path + file_name, accumulated_face_data)
            print("Dataset saved at : {}".format(dataset_path + file_name + '.npy'))
            accumulated_face_data = []  # Clear accumulated data



            ret, jpeg = cv.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    except Exception as e:
        print(f"Error in generate: {e}")
        yield HttpResponseServerError("Video source error")
    finally:
        if video_capture:
            video_capture.release()

def video_feed(request):
    return StreamingHttpResponse(generate(request.GET.get('username','')), content_type='multipart/x-mixed-replace; boundary=frame')

def faceAuth():
    def distances (v1, v2):
        return np.sqrt(((v1-v2)**2).sum())

    def knn(train, test, k=5):
        dist=[]

        for i in range (train.shape[0]):
            ix=train[i,:-1]
            iy=train[i, -1]

            d=distances(test, ix)
            dist.append([d, iy])

        dk= sorted(dist, key=lambda x: x[0])[:k]
        labels= np.array(dk)[:,-1]

        output=np.unique(labels, return_counts=True)

        index= np.argmax(output[1])
        return output[0][index]

    try:
        cap = cv.VideoCapture(0, cv.CAP_DSHOW)

        if not cap.isOpened():
            raise RuntimeError('Could not open video source.')
        face_cascade= cv.CascadeClassifier("E:/_Coding/openCV/Projects/real_world_projects/Login-SignUp_Using_Face_Authentication/opencvtrystream/opencvstreamapp/static/haarcascade_frontalface_alt.xml")

        skip=0
        dataset_path="E:/_Coding/openCV/Projects/real_world_projects/Login-SignUp_Using_Face_Authentication/opencvtrystream/opencvstreamapp/static/face_datasets/"

        face_data=[]
        labels=[]
        class_id=0
        names={}

        for fx in os.listdir(dataset_path):
            if fx.endswith('.npy'):
                names[class_id]=fx[:-4]
                data_item = np.load(dataset_path + fx)
                face_data.append(data_item)

                target=class_id * np.ones((data_item.shape[0],))
                class_id+=1
                labels.append(target)

        face_dataset= np.concatenate(face_data, axis=0)
        face_labels=np.concatenate(labels, axis=0).reshape((-1,1))
        print(face_labels.shape)
        print(face_dataset.shape)

        trainset=np.concatenate((face_dataset, face_labels), axis=1)
        print(trainset.shape)

        font=cv.FONT_HERSHEY_SIMPLEX

        while True:
            ret, frame=cap.read()
            if ret== False:
                continue

            gray= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            faces=face_cascade.detectMultiScale(gray, 1.3,5)

            for face in faces:
                x,y,w,h = face

                offset=5

                face_section= frame[y-offset: y+h+offset, x-offset: x+w+offset]
                face_section=cv.resize(face_section,(100,100))

                out=knn(trainset, face_section.flatten())

                cv.putText(frame, names[int(out)], (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv.LINE_AA)
                cv.rectangle(frame, (x,y), (x+w, y+h), (255, 255,255),2)


            # cv.imshow("Recognizing Face", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break


            retur, jpeg= cv.imencode('.jpg', frame)
            frame_bytes= jpeg.tobytes()
            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
    except Exception as e:
        print(f"Error in generate: {e}")
        yield HttpResponseServerError("Video source error")
    finally:
        if cap:
            cap.release()



def open_camera(request):
    return StreamingHttpResponse(faceAuth(), content_type='multipart/x-mixed-replace; boundary=frame')









# def capture_video(request):
#     return render(request, 'face_capture.html')


# def capture_face_data(username):

#     cap= cv.VideoCapture(0)
#     face_cascade= cv.CascadeClassifier("E:/_Coding/openCV/face detection & recognition/haarcascade_frontalface_alt.xml")

#     skip=0
#     face_data=[]
    # dataset_path="E:/_Coding/openCV/face detection & recognition/face_dataset/"

    # file_name= input("Enter the name of person: ")

    # while True:
    #     ret,frame=cap.read()

    #     gray_frame=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #     if ret==False:
    #         continue
    #     faces=face_cascade.detectMultiScale(gray_frame, 1.3,5)
    #     if len(faces)==0:
    #         continue
        # k=1
        # faces= sorted (faces, key =lambda x:x[2]*x[3], reverse=True)
        # skip+=1
        # for face in faces[:1]:
        #     x,y,w,h= face
        #     offset=5
        #     face_offset = frame[y-offset: y+h+offset,x-offset:x+w+offset]
        #     face_selection= cv.resize(face_offset, (100,100))
        #     if skip % 10==0:
        #         face_data.append(face_selection)
        #         print(len(face_data))
        #     cv.imshow(str(k),face_selection)
        #     k+=1

    #         cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    #     cv.imshow("Faces", frame)

    #     if cv.waitKey(1) & 0xFF==ord('q'):
    #         break
    # face_data= np.array(face_data)
    # face_data=face_data.reshape((face_data.shape[0],-1))
    # print(face_data.shape)


    # cap.release()
    # cv.destroyAllWindows()

    # return face_data

# def capture_face_data_ajax(request):
#     if request.method == 'POST':
#         username = request.POST.get('username')

#         # Capture face data
#         face_data = capture_face_data(username)

#         # Return response to client (e.g., success message or error)
#         return JsonResponse({'status': 'success', 'message': 'Face data captured'})
