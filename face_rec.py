import random
from datetime import date, datetime
import face_recognition
import imutils
import pickle
import cv2
import os
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

# find path of xml file containing haarcascade file
cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)
# load the known faces and embeddings saved in last file
data = pickle.loads(open('face_enc', "rb").read())
recognized_faces = {}


class VideoPlayer:
    def __init__(self, master=None, width=100, height=100):
        self.cap = cv2.VideoCapture(0)
        self.master = master
        self.canvas = tk.Canvas(master, height=height, width=width)
        self.delay = int(1000 / self.cap.get(cv2.CAP_PROP_FPS))
        self.face_locations = ''
        self.names = []
        self.process_this_frame = True
        self.unknown_faces = []

    def place(self, x, y):
        self.canvas.place(x=x, y=y)
        self.update()

    def update(self):
        if button['text'] == 'Pause':
            ret, frame = self.cap.read()
        else:
            self.master.after(self.delay, self.update)
            return
        if ret:
            if button['text'] == 'Pause':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.process_this_frame:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.375, fy=0.375)

                    # the facial embeddings for face in input
                    self.face_locations = face_recognition.face_locations(small_frame)
                    face_encodings = face_recognition.face_encodings(small_frame, self.face_locations)
                    self.names = []

                    # loop over the facial embeddings incase
                    # we have multiple embeddings for multiple fcaes
                    for encoding in face_encodings:
                        # Compare encodings with encodings in data["encodings"]
                        # Matches contain array with boolean values and True for the embeddings it matches closely
                        # and False for rest
                        matches = face_recognition.compare_faces(data["encodings"], encoding)
                        # set name unknown if no encoding matches
                        name = "Unknown"
                        # check to see if we have found a match
                        if True in matches:
                            # Find positions at which we get True and store them
                            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                            counts = {}
                            # loop over the matched indexes and maintain a count for
                            # each recognized face
                            for i in matchedIdxs:
                                # Check the names at respective indexes we stored in matchedIdxs
                                name = data["names"][i]
                                # increase count for the name we got
                                counts[name] = counts.get(name, 0) + 1
                            # set name which has highest count
                            name = max(counts, key=counts.get)

                        if name == 'Unknown':
                            data['encodings'].append(encoding)
                            count_unknown = random.randint(1, 100)
                            while count_unknown in self.unknown_faces:
                                count_unknown = random.randint(1, 100)
                            self.unknown_faces.append(count_unknown)
                            data['names'].append(f'Unknown_{count_unknown}')


                        # update the list of names
                        self.names.append(name)
                self.process_this_frame = not self.process_this_frame
                    # loop over the recognized faces
                for (top, right, bottom, left), name in zip(self.face_locations, self.names):
                    name: str
                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                    top = int(top * 8 / 3)
                    right = int(right * 8 / 3)
                    bottom = int(bottom * 8 / 3)
                    left = int(left * 8 / 3)
                    # rescale the face coordinates
                    # draw the predicted face name on the image
                    if name[:7] == 'Unknown':
                        if name not in recognized_faces:
                            recognized_faces[name] = str(datetime.now())
                        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                        cv2.putText(frame, name, (left + 6, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.75, (255, 255, 255), 1)
                    else:
                        if name not in recognized_faces:
                            recognized_faces[name] = str(datetime.now())
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, name, (left + 6, bottom + 20), cv2.FONT_HERSHEY_DUPLEX,
                                    0.75, (255, 255, 255), 1)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.master.after(self.delay, self.update)
        else:
            self.cap.release()


def pause_unpause():
    if button['text'] == 'Pause':
        button['text'] = 'Play'
    else:
        button['text'] = 'Pause'


def stop():
    button['text'] = 'Play'
    button['state'] = tk.DISABLED
    print(recognized_faces)


window = tk.Tk()
window.geometry('650x550')
button = tk.Button(window, text='Play', command=pause_unpause)
button.place(x=0, y=500)
button1 = tk.Button(window, text='Stop', command=stop)
button1.place(x=100, y=500)
video = VideoPlayer(master=window, width=650, height=500)
video.place(x=0, y=0)
window.mainloop()

