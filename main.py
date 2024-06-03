import tkinter as tk
from tkinter import messagebox
import requests
from PIL import Image, ImageTk
import pickle
import cv2
import os
import face_recognition
import random
from datetime import datetime


# find path of xml file containing haarcascade file
cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)
# load the known faces and embeddings saved in last file
data = pickle.loads(open('face_enc', "rb").read())
recognized_faces = {}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Многостраничное приложение")
        self.geometry("800x600")

        # Создаем контейнер для смены окон
        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)

        self.frames = {}

        for F in (LoginPage, EventSelectionPage, FaceRecognitionPage):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("LoginPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


class LoginPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.label = tk.Label(self, text="Авторизация")
        self.label.pack(pady=10)

        self.username_label = tk.Label(self, text="Логин")
        self.username_label.pack(pady=5)
        self.username_entry = tk.Entry(self)
        self.username_entry.pack(pady=5)

        self.password_label = tk.Label(self, text="Пароль")
        self.password_label.pack(pady=5)
        self.password_entry = tk.Entry(self, show="*")
        self.password_entry.pack(pady=5)

        self.login_button = tk.Button(self, text="Войти", command=self.login)
        self.login_button.pack(pady=20)

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        # Отправка запроса на сервер
        response = requests.post("http://facial-rec.com/api/login", data={"username": username, "password": password})

        if response.status_code == 200:
            self.controller.show_frame("EventSelectionPage")
        else:
            tk.messagebox.showerror("Ошибка", "Неверный логин или пароль")


class EventSelectionPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.label = tk.Label(self, text="Выбор мероприятия")
        self.label.pack(pady=10)

        self.events_listbox = tk.Listbox(self)
        self.events_listbox.pack(pady=10)

        self.download_button = tk.Button(self, text="Скачать файл", command=self.download_file)
        self.download_button.pack(pady=20)

        # Получение списка мероприятий (пример)
        self.load_events()

    def load_events(self):
        # Замените URL на ваш
        response = requests.get(f"http://facial-rec.com/api/events/")

        if response.status_code == 200:
            events = response.json()
            for event in events:
                self.events_listbox.insert(tk.END, event["name"])

    def download_file(self):
        selected_event = self.events_listbox.get(tk.ACTIVE)
        # Замените URL на ваш
        response = requests.get(f"http://facial-rec/api/events/{selected_event}/download")

        if response.status_code == 200:
            with open("face_enc", "wb") as f:
                f.write(response.content)
            self.controller.show_frame("FaceRecognitionPage")
        else:
            tk.messagebox.showerror("Ошибка", "Не удалось скачать файл")


class FaceRecognitionPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.label = tk.Label(self, text="Распознавание лиц")
        self.label.pack(pady=10)

        # Вставляем ваш существующий код для работы с камерой и распознавания лиц
        self.button = tk.Button(self, text='Play', command=self.pause_unpause)
        self.button.pack(side="left")

        self.button1 = tk.Button(self, text='Stop', command=self.stop)
        self.button1.pack(side="right")

        self.video = VideoPlayer(master=self, width=650, height=500)
        self.video.place(x=0, y=0)

    def pause_unpause(self):
        if self.button['text'] == 'Pause':
            self.button['text'] = 'Play'
        else:
            self.button['text'] = 'Pause'

    def stop(self):
        self.button['text'] = 'Play'
        self.button['state'] = tk.DISABLED
        self.send_data()
        print(recognized_faces)

    def send_data(self):
        event_name = self.controller.selected_event
        url = "http://facial-rec.com/api/upload_faces"

        files = {"file": ("recognized_faces.pkl", pickle.dumps(recognized_faces))}
        data = {"event": event_name}

        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            messagebox.showinfo("Успех", "Данные успешно отправлены")
        else:
            messagebox.showerror("Ошибка", "Не удалось отправить данные")


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
        if self.master.button['text'] == 'Pause':
            ret, frame = self.cap.read()
        else:
            self.master.after(self.delay, self.update)
            return
        if ret:
            if self.master.button['text'] == 'Pause':
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
                    if name[:8] == 'Unknown_':
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


if __name__ == "__main__":
    app = App()
    app.mainloop()
