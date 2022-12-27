import cv2
import numpy as np
import mediapipe as mp
from spellchecker import SpellChecker
import translators as ts
from keras.models import load_model
from tkinter import *
import tkinter
import PIL.Image
import PIL.ImageTk
import time

model = load_model('D:/Model ASL/ASL landmarks using Dense v2.h5')

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space']
str = ""
spell = SpellChecker()
flag = False
flag2 = False
Lang = True
Trans = False
correction = ""


def mediapipe_detection(image, model):
    # COLOR CONVERSION BGR 2 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
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


class App:
    global holistic

    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title = (window_title)
        self.video_source = video_source

        self.vid = MyVideoCapture(self.video_source)
        self.canvas = tkinter.Canvas(
            window, width=self.vid.width - 4, height=self.vid.height + 280)
        self.canvas.pack()

        text_frame = tkinter.Frame(
            window, background=self.from_rgb((17, 24, 31)))
        text_frame.place(x=0, y=self.vid.height+50, anchor="nw",
                         width=self.vid.width, height=208)

        self.text = tkinter.Text(
            text_frame, state='disable', width=115, height=19, font=("Times New Roman", 16), bg="#C0C0C0")
        self.text.pack()

        # scroll bar
        scrollbar = Scrollbar(self.text)
        scrollbar.place(relheight=1, relx=0.99)
        scrollbar.configure(command=self.text.yview)

        btn_frametop = tkinter.Frame(
            window, background=self.from_rgb((117, 123, 129)))
        btn_frametop.place(x=0, y=0, anchor="nw", width=self.vid.width)

        btn_framebot = tkinter.Frame(
            window, background=self.from_rgb((117, 123, 129)))
        btn_framebot.place(x=0, y=self.vid.height,
                           anchor="nw", width=self.vid.width)

        self.btn_Viet = tkinter.Button(
            btn_frametop, text="Tieng Viet", width=10, command=self.Language_Viet, font=FONT_BOLD, bg=BG_GRAY)
        self.btn_Viet.pack(side="left", padx=10, pady=10)

        self.btn_Eng = tkinter.Button(
            btn_frametop, text="English", width=10, command=self.Language_Eng, font=FONT_BOLD, bg=BG_GRAY)
        self.btn_Eng.pack(side="left", padx=10, pady=10)

        self.btn_Trans = tkinter.Button(
            btn_framebot, text="Translate", width=10, command=self.Translate, font=FONT_BOLD, bg=BG_GRAY)
        self.btn_Trans.pack(side="left", padx=10, pady=10)

        self.btn_Correct = tkinter.Button(
            btn_framebot, text="Correction", width=10, command=self.Correction, font=FONT_BOLD, bg=BG_GRAY)
        self.btn_Correct.pack(side="right", padx=10, pady=10)

        self.btn_results = tkinter.Button(
            btn_frametop, text="Detect", width=10, command=self.get_results, font=FONT_BOLD, bg=BG_GRAY)
        self.btn_results.pack(side="right", padx=10, pady=10)
        self.window.bind("<space>", lambda event=None: self.get_results())

        self.btn_delete = tkinter.Button(
            btn_frametop, text="Delete All Text", width=15, command=self.delete_text, font=FONT_BOLD, bg=BG_GRAY)
        self.btn_delete.pack(side="right", padx=10, pady=10)

        self.delay = 15
        self.update()

        self.window.mainloop()

    def update_text(self):
        global str
        self.text.configure(state="normal")
        self.text.delete(1.0, END)
        self.text.insert('end', str)
        self.text.configure(state='disable')

    def delete_text(self):
        global str
        str = ''
        self.text.configure(state="normal")
        self.text.delete(1.0, END)
        self.text.configure(state='disable')

    def Language_Viet(self):
        global Lang
        Lang = False
        self.text.configure(state="normal")
        self.text.insert(
            'end', '\n' + "Da chon tieng Viet")
        self.text.configure(state='disable')

    def Language_Eng(self):
        global Lang
        Lang = True
        self.text.configure(state="normal")
        self.text.insert(
            'end', '\n' + "Selected English")
        self.text.configure(state='disable')

    def update(self):
        ret, frame = self.vid.get_frame(holistic)

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

            self.window.after(self.delay, self.update)

    def Correction(self):
        global str
        global flag
        global Lang
        global correction
        if not Lang:
            self.text.configure(state="normal")
            self.text.insert(
                'end', '\n' + "Chỉ sửa lỗi chính tả đối với tiếng anh")
            self.text.configure(state='disable')
        elif flag:
            str = correction
            str += ' '
            flag = False
            self.update_text()

    def Translate(self):
        global Lang
        global flag2
        if not Lang:
            self.text.configure(state="normal")
            self.text.insert(
                'end', '\n' + "If you want to translate text to VietNamese please select English as input")
            self.text.configure(state='disable')
        elif not flag2:
            self.text.configure(state="normal")
            self.text.insert(
                'end', '\n' + "You have to finnish the sentece in other to translate")
            self.text.configure(state='disable')
        else:
            res = ts.google(str, to_language='vi')
            flag2 = False
            self.text.configure(state="normal")
            self.text.insert('end', '\n' + res)
            self.text.configure(state='disable')
        flag2 = False

    def get_results(self):
        global results
        global str
        global flag2
        global Lang
        global flag
        global spell
        global correction
        global Trans
        if results.left_hand_landmarks == None and results.right_hand_landmarks == None:
            self.text.configure(state="normal")
            self.text.insert('end', '\n' + "No Hands Detect Yet")
            self.text.configure(state='disable')
        else:
            # Extract keypoints
            keypoints = extract_keypoints(results)
            keypoints = keypoints.reshape(-1, 63)

            # Make prediction
            prediction = np.argmax(model.predict(keypoints)[0])
            index = letterpred[prediction]
            if prediction != 26 and prediction != 27:
                str += index
                flag = False
                flag2 = False
            else:
                if prediction == 27:
                    if Lang:
                        flag2 = True
                        correction = ' '.join(
                            [spell.correction(word) for word in str.split()]).upper()
                    if correction != str and Lang:
                        flag = True
                    if not flag and Lang:
                        Trans = True
                    str += ' '
                elif prediction == 26:
                    str = str[:-1]
            self.update_text()
            if flag:
                self.text.configure(state="normal")
                self.text.insert('end', '\n' + "Do you mean: " + correction)
                self.text.configure(state='disable')
            if flag2:
                self.text.configure(state="normal")
                self.text.insert(
                    'end', '\n' + "Do you want to translate to Vietnamese: ")
                self.text.configure(state='disable')

    def from_rgb(self, rgb):
        return "#%02x%02x%02x" % rgb


class MyVideoCapture:
    """docstring for  MyVideoCapture"""

    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("unable open video source", video_source)

        self.vid.set(3, 1280)
        self.vid.set(4, 720)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self, holistic):
        global results
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            image, results = mediapipe_detection(frame, holistic)
            if results.left_hand_landmarks == None and results.right_hand_landmarks == None:
                pass
            else:
                # Draw a box around hand
                draw_border(image, results)

                # Draw landmarks
                draw_styled_landmarks(image, results)

            if ret:
                return (ret, image)
            else:
                return (ret, None)
        else:
            return (ret, None)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    App(tkinter.Tk(), "tkinter ad OpenCV")
