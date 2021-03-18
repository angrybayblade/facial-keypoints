import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

with open("model/model.json","r") as model:
    model = tf.keras.models.model_from_json(model.read())
    model.load_weights("./model/weights")

def get_image(cam:cv2.VideoCapture)->np.ndarray:
    _,frame = cam.read()
    return frame

cam = cv2.VideoCapture(0)
cas = cv2.CascadeClassifier()

cas.load("./cas.xml")

oneThird = 1/3
oneForth = 1/4
oneFifth = 1/5
oneSixth = 1/6

while True:
    frame = get_image(cam)
    gray = frame.mean(axis=-1).astype(np.uint8)
    face = cas.detectMultiScale(gray)

    if len(face):
        (xmin,ymin,w,h),*_ = face
        xmax = xmin + w
        ymax = ymin + h + int(h * oneSixth )
        ymin = ymin - int(h * oneSixth )
        frame = cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(255,0,0),1)
        
        cords = (96 * model.predict(cv2.resize(gray[ymin:ymax,xmin:xmax],(96,96)).reshape(-1,96,96,1))).astype(int).reshape(15,2)
        face = cv2.resize(frame[ymin:ymax,xmin:xmax],(96,96))
        for x,y in cords:
            face = cv2.circle(face,(x,y),1,(255,0,0),1)     

        cv2.imshow("face",cv2.resize(face,(256,256)))

    cv2.imshow("frame",frame[:,::-1,:])
    

    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
