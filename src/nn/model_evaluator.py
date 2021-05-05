import pickle
import numpy as np
import cv2
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from tensorflow_core.python.keras import Sequential

from src.data_util.data_cleanser import clean
from src.video_tag import write_on_frame

input_video_path = '/Users/hbojja/uiuc/CS445-CP/FinalProject/input/tape_1.mov'

cap = cv2.VideoCapture(input_video_path)

model_base_path = '/Users/hbojja/uiuc/CS445-CP/FinalProject/trained_models'

model:Sequential = tf.keras.models.load_model(model_base_path + '/model')

lb:LabelBinarizer = pickle.load(open(model_base_path + '/vec', 'rb'))
scaler:MinMaxScaler = pickle.load(open(model_base_path + '/scaler', 'rb'))

last_rec_label = ''

frame_count = 0
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_count = frame_count + 1
    if frame_count % 4 != 0:
        continue

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    resized = clean(gray).ravel()
    img = np.array([resized])
    img = scaler.transform(img)

    print(img.shape)
    predictions = model.predict(img)

    predictions_classes = np.where(predictions > 0.7, 1, 0)

    label = lb.inverse_transform(predictions_classes)[0]
    label_to_write = 'other'
    # Display the resulting frame
    if last_rec_label == label:
        label_to_write = label

    write_on_frame(frame, label_to_write)

    last_rec_label = label
    #
    # count = count + 1
    # last_rec_label = rec_label
    #
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
