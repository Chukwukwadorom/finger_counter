import cv2
from HandTrackModule import HandDetection
import time
import os

cap = cv2.VideoCapture(0)

hand_detection = HandDetection(min_detection_con=0.8)

prevTime = 0


finger_tips = [4,8,12,16,20]

pics = os.listdir("pictures")

pic_list = []

for pix in pics:
    pic_list.append(cv2.resize(cv2.imread(os.path.join("pictures",pix)), (120, 150)))
    
while True:
    success, frame = cap.read()

    curTime = time.time()
    fps = int(round(1/(curTime - prevTime)))
    prevTime=curTime


    if not success:
        print("could not read frame")
        break

    frame = cv2.flip(frame, 1)
    frame = hand_detection.get_hands(frame)
    position, hand_type = hand_detection.get_positions(frame)
    tip_up = []
    if len(position) > 0:

        if (hand_type) == "Right":
            if  position[finger_tips[0]][1] < position[finger_tips[0]-1][1]:
                tip_up.append(1)
            else:
                tip_up.append(0)
        elif (hand_type) == "Left":
            if position[finger_tips[0]][1] > position[finger_tips[0]-1][1]:
                tip_up.append(1)
            else:
                tip_up.append(0)


        for id in finger_tips[1:]:
            if position[id][2] < position[id-1][2]:
                tip_up.append(1)
            else:
                tip_up.append(0)

        print(tip_up)
        fingers_up = sum(tip_up, 0)
        finger_pix = pic_list[fingers_up]
        h, w, c = finger_pix.shape

        frame[40:40+h, 5:5+w, :] = finger_pix


        frame = cv2.rectangle(frame, (5,350), (100, 450), (0,215,255), -1)
        frame = cv2.putText(frame, f'{fingers_up}', (45, 410), cv2.FONT_HERSHEY_SIMPLEX,
                         1, (78.4, 47.1, 31.4), 3)

    frame = cv2.putText(frame, f'FPS: {fps}', (5,30), cv2.FONT_HERSHEY_PLAIN,
                        1, (200, 0, 0), 2)
    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

