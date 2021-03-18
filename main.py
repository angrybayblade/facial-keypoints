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

xmin = 224
xmax = 416
ymin = 144
ymax = 336


while True:
    frame = get_image(cam)[ymin:ymax,xmin:xmax]
    cords = (192 * model.predict(
            cv2.resize(
                    frame.mean(axis=-1).astype(np.uint8),
                    (96,96)
                ).reshape(-1,96,96,1)/255
            )
        ).astype(int).reshape(15,2)

    for x,y in cords:
        frame = cv2.circle(frame,(x,y),1,(255,0,0),1)     

    cv2.imshow("frame",frame)

    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
