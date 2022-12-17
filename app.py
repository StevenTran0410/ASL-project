from flask import Flask, request, render_template, Response
import cv2
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
from spellchecker import SpellChecker
import translators as ts
from keras.models import load_model

app = Flask(__name__)
cap = cv2.VideoCapture(0)

model = load_model('D:\Model ASL\ASL landmarks using Dense v2.h5')

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
mp_hands = mp.solutions.hands


def mediapipe_detection(image, model):
    # COLOR CONVERSION BGR 2 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_styled_landmarks(image, results):
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


def draw_border(image, results):
    h, w, c = image.shape
    if results.left_hand_landmarks:
        hand_landmarks = [results.left_hand_landmarks]
    elif results.right_hand_landmarks:
        hand_landmarks = [results.right_hand_landmarks]
    else:
        hand_landmarks = False

    x_max = 0
    y_max = 0
    x_min = w
    y_min = h

    if hand_landmarks:
        for handLMs in hand_landmarks:
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            cv2.rectangle(image, (x_min, y_min),
                          (x_max, y_max), (255, 0, 105), 2)


def extract_keypoints(results):
    if results.left_hand_landmarks != None:
        x = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
        ) if results.left_hand_landmarks else np.zeros(21*3)
    elif results.right_hand_landmarks != None:
        x = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
        ) if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([x])


def processForm():
    text = request.form("select")
    return text


def generate_frames():
    letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space']
    with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
        while True:

            # read the camera frame
            cap.set(3, 1280)
            cap.set(4, 720)
            success, frame = cap.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                show = buffer.tobytes()
                image, results = mediapipe_detection(frame, holistic)
                if results.left_hand_landmarks == None and results.right_hand_landmarks == None:
                    index = 'Nothing'
                else:
                    # Draw a box around hand
                    draw_border(image, results)
                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    ret, buffer = cv2.imencode('.jpg', image)
                    show = buffer.tobytes()

                    # Extract keypoints
                    keypoints = extract_keypoints(results)
                    keypoints = keypoints.reshape(-1, 63)

                    # Make prediction
                    prediction = np.argmax(model.predict(keypoints)[0])
                    index = letterpred[prediction]
                    print(index)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + show + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
