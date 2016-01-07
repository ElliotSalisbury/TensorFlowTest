import pickle
import numpy as np
import cv2

videoPath = "vanuatu35.mp4"
capture = cv2.VideoCapture(videoPath)

fourcc = cv2.cv.CV_FOURCC(*'XVID')
writer = cv2.VideoWriter("output_vanuatu35.avi",fourcc,30,(1280,360))
ret, frame = capture.read()
frameId = 0
while ret:
    frame = cv2.resize(frame, (640,360))
    height,width,depth = frame.shape

    with open('maskFrames/%06d.pickle'%frameId, 'rb') as f:
        mask = pickle.load(f)

    mask = cv2.resize(mask, (640,360), interpolation=cv2.INTER_NEAREST)
    mask *= 255
    mask = mask.astype(np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    outFrame = np.zeros((360,1280,3), dtype=np.uint8)
    outFrame[0:360,0:640] = frame
    outFrame[0:360,640:1280] = mask

    writer.write(outFrame)

    cv2.imshow("mask",outFrame)
    cv2.waitKey(1)

    ret, frame = capture.read()
    frameId += 1

writer.release()
capture.release()
cv2.destroyAllWindows()