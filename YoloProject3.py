import cv2
import numpy as np
import time

# Input model YOLO
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")
classes = []
with open("classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
count = 0

#Input Video
cap = cv2.VideoCapture('sampah2.mp4')
font = cv2.FONT_HERSHEY_SIMPLEX
_, frame = cap.read()
height, width, channels = frame.shape

'''#Input Video
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

timeframe = time.time()
frame_id = 0

while True:
  _, frame = cap.read()
  frame_id += 1
  height, width, channels = frame.shape'''

corner_y =int(height/8) - 30
corner_x = int(width/2) + 30
timeframe = time.time()
frame_id = 0
label_before = ''
counting = ''

while True:
  _, frame = cap.read()
  frame_id += 1
  height, width, channels = frame.shape

  # Deteksi Objek
  blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
  net.setInput(blob)
  outs = net.forward(output_layers)

  # Menampilkan info di Screen
  class_ids = []
  confidences = []
  boxes = []
  for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

  indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

  for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        #color = colors[i]
        color = [255,255,255]
        print('Sebelum %s Sesudah %s' % (label_before,label))
        if label_before == 'Pembawa Sampah' and label == 'Orang':

            cv2.rectangle(frame, (x, y), (x + w, y + h), [220,20,60], 2)
            cv2.putText(frame, 'WARNING', (x, y + 30), font, 1, [220,20,60], 2)
            count += 1
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 1, color, 2)
        label_before = label
  if count > 0:
      counting = 'ADA PEMBUANG SAMPAH'
  counting = 'ADA PEMBUANG SAMPAH : %d' % count
  cv2.putText(frame, counting, (corner_x, corner_y ), font, 0.5, [220, 20, 60], 2)
  cv2.imshow("image", frame)
  key = cv2.waitKey(1)
  if key == 27:
    break

cap.release()
cv2.destroyAllWindows()