import numpy as np
import cv2

from src.traffic_sign_recogninzer import traffic_sign_recogninzer
from src.video_tag import write_on_frame

input_video_path = '/Users/hbojja/uiuc/CS445-CP/FinalProject/input/tape_2.mov'

cap = cv2.VideoCapture(input_video_path)

count = 0
traffic_sign_rec = traffic_sign_recogninzer()

last_rec_label = ''

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rec_label, fair_des_count = traffic_sign_rec.realtime_template_matcher(gray)

    if last_rec_label == rec_label:
        label = rec_label + '(' + str(fair_des_count) + ')'

    # Display the resulting frame
    write_on_frame(frame, label + str(count))

    count = count + 1
    last_rec_label = rec_label

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
