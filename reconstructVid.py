import pickle
import numpy as np
import cv2

videoPath = "vanuatu35.mp4"
capture = cv2.VideoCapture(videoPath)

fourcc = cv2.cv.CV_FOURCC(*'XVID')
writer = cv2.VideoWriter("output_vanuatu35.avi",fourcc,30,(1280,360))
ret, frame = capture.read()
frameId = 0

frameSize = (640,360)
minFrameSize = (64,36)
numOfSizes = 10
frameWidths = list(reversed(range(minFrameSize[0],frameSize[0]+1,frameSize[0]/numOfSizes)))
frameHeights = list(reversed(range(minFrameSize[1],frameSize[1]+1,frameSize[1]/numOfSizes)))

windowSize=28

largeLocations = [
    [sum(frameWidths[:0]),0,sum(frameWidths[:1]),frameHeights[0]],
    [sum(frameWidths[:1]),0,sum(frameWidths[:2]),frameHeights[1]],
    [sum(frameWidths[:2]),0,sum(frameWidths[:3]),frameHeights[2]],
    [sum(frameWidths[3:3]),frameHeights[0],sum(frameWidths[3:4]),frameHeights[0]+frameHeights[3]],
    [sum(frameWidths[3:4]),frameHeights[0],sum(frameWidths[3:5]),frameHeights[0]+frameHeights[4]],
    [sum(frameWidths[3:5]),frameHeights[0],sum(frameWidths[3:6]),frameHeights[0]+frameHeights[5]],
    [sum(frameWidths[3:6]),frameHeights[0],sum(frameWidths[3:7]),frameHeights[0]+frameHeights[6]],
    [sum(frameWidths[3:7]),frameHeights[0],sum(frameWidths[3:8]),frameHeights[0]+frameHeights[7]],
    [sum(frameWidths[3:8]),frameHeights[0],sum(frameWidths[3:9]),frameHeights[0]+frameHeights[8]],
    [sum(frameWidths[3:8]),frameHeights[0]+frameHeights[7],sum(frameWidths[3:8])+frameWidths[9],frameHeights[0]+frameHeights[7]+frameHeights[9]]]

ret, frame = capture.read()
frameId = 1
while ret:
  frame = cv2.resize(frame, (frameSize[0],frameSize[1]))
  height,width,depth = frame.shape

  with open('resultsPerSize/%06d.pickle'%frameId, 'rb') as f:
    resultsPerSize = pickle.load(f)

  bigMask = np.zeros((frameHeights[0]+frameHeights[3],sum(frameWidths[:3]),3),dtype=np.uint8)
  masks = np.zeros((10,frameHeights[0],frameWidths[0]),dtype=np.float64)
  for i in range(numOfSizes):
    fWidth = frameWidths[i]
    fHeight = frameHeights[i]
    maskCols = len(range(0,fWidth-windowSize,windowSize/2))
    maskRows = len(range(0,fHeight-windowSize,windowSize/2))

    mask = np.zeros((maskRows,maskCols),dtype=np.float64)
    for j, result in enumerate(resultsPerSize[i]):
      y = j % maskRows
      x = j / maskRows

      mask[y,x] = result[1]

    mask = cv2.resize(mask, (frameWidths[0],frameHeights[0]), interpolation=cv2.INTER_LINEAR)
    smallMask = cv2.resize(mask, (frameWidths[i],frameHeights[i]), interpolation=cv2.INTER_LINEAR)
    smallMask *= 255
    smallMask = smallMask.astype(np.uint8)
    smallMask = cv2.cvtColor(smallMask, cv2.COLOR_GRAY2BGR)

    minX = largeLocations[i][0]
    minY = largeLocations[i][1]
    maxX = largeLocations[i][2]
    maxY = largeLocations[i][3]
    bigMask[minY:maxY,minX:maxX] = smallMask
    masks[i,0:frameHeights[0],0:frameWidths[0]] = mask

  maxMask = masks[0:9].mean(axis=0)
  maxMask *= 255
  maxMask = maxMask.astype(np.uint8)
  maxMask = cv2.cvtColor(maxMask, cv2.COLOR_GRAY2BGR)

  outFrame = np.zeros((frameHeights[0],frameWidths[0]*2,3), dtype=np.uint8)
  outFrame[0:frameHeights[0],0:frameWidths[0]] = frame
  outFrame[0:frameHeights[0],frameWidths[0]:frameWidths[0]*2] = maxMask

  writer.write(outFrame)

  cv2.imshow("outFrame", outFrame)
  cv2.waitKey(1)

  ret, frame = capture.read()
  frameId += 1

writer.release()
capture.release()
cv2.destroyAllWindows()