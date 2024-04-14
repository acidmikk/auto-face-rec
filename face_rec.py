import face_recognition
import imutils
import pickle
import time
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

# video_capture = cv2.VideoCapture(0)
# # loop over frames from the video file stream
# while True:
#     # grab the frame from the threaded video stream
#     ret, frame = video_capture.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(gray,
#                                          scaleFactor=1.1,
#                                          minNeighbors=5,
#                                          minSize=(60, 60),
#                                          flags=cv2.CASCADE_SCALE_IMAGE)
#
#     # convert the input frame from BGR to RGB
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     # the facial embeddings for face in input
#     encodings = face_recognition.face_encodings(rgb)
#     names = []
#     # loop over the facial embeddings incase
#     # we have multiple embeddings for multiple fcaes
#     for encoding in encodings:
#         # Compare encodings with encodings in data["encodings"]
#         # Matches contain array with boolean values and True for the embeddings it matches closely
#         # and False for rest
#         matches = face_recognition.compare_faces(data["encodings"],
#                                                  encoding)
#         # set name =inknown if no encoding matches
#         name = "Unknown"
#         # check to see if we have found a match
#         if True in matches:
#             # Find positions at which we get True and store them
#             matchedIdxs = [i for (i, b) in enumerate(matches) if b]
#             counts = {}
#             # loop over the matched indexes and maintain a count for
#             # each recognized face
#             for i in matchedIdxs:
#                 # Check the names at respective indexes we stored in matchedIdxs
#                 name = data["names"][i]
#                 # increase count for the name we got
#                 counts[name] = counts.get(name, 0) + 1
#             # set name which has highest count
#             name = max(counts, key=counts.get)
#
#         # update the list of names
#         names.append(name)
#         # loop over the recognized faces
#         for ((x, y, w, h), name) in zip(faces, names):
#             # rescale the face coordinates
#             # draw the predicted face name on the image
#             if name == 'Unknown':
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#                 cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
#                             0.75, (0, 0, 255), 2)
#             else:
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.75, (0, 255, 0), 2)
#     cv2.imshow("Frame", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# video_capture.release()
# cv2.destroyAllWindows()


class VideoPlayer:
    def __init__(self, master=None, width=100, height=100):
        self.cap = cv2.VideoCapture(0)
        self.master = master
        self.canvas = tk.Canvas(master, height=height, width=width)
        self.delay = int(1000 / self.cap.get(cv2.CAP_PROP_FPS))

        self.process_this_frame = True

    def place(self, x, y):
        self.canvas.place(x=x, y=y)
        self.update()

    def update(self):
        if button['text'] == 'stop':
            ret, frame = self.cap.read()
        else:
            self.master.after(self.delay, self.update)
            return
        if ret:
            if button['text'] == 'stop':
                if self.process_this_frame:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

                    # convert the input frame from BGR to RGB
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    # the facial embeddings for face in input
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    names = []
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

                        # update the list of names
                        names.append(name)
                        # loop over the recognized faces
                        for (top, right, bottom, left), name in zip(face_locations, names):
                            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                            top *= 2
                            right *= 2
                            bottom *= 2
                            left *= 2
                            # rescale the face coordinates
                            # draw the predicted face name on the image
                            if name == 'Unknown':
                                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                                cv2.putText(frame, name, (left + 6, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.75, (255, 255, 255), 1)
                            else:
                                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                                cv2.putText(frame, name, (left + 6, bottom + 20), cv2.FONT_HERSHEY_DUPLEX,
                                            0.75, (255, 255, 255), 1)
                    self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.process_this_frame = not self.process_this_frame
            self.master.after(self.delay, self.update)
        else:
            self.cap.release()


def pause_unpause():
    if button['text'] == 'stop':
        button['text'] = 'play'
    else:
        button['text'] = 'stop'


window = tk.Tk()
window.geometry('650x550')
button = tk.Button(window, text='stop', command=pause_unpause)
button.place(x=0, y=500)
video = VideoPlayer(master=window, width=650, height=500)
video.place(x=0, y=0)


window.mainloop()
