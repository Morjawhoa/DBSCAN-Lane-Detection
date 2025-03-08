import cv2
import time
from Controller import PhotoGrapher, Messenger, key_to_command
from detect_keyboard_keys import key_check

camera_url = 'http://192.168.1.1:8080/?action=snapshot'
control_host = '192.168.1.1'
control_port = 2001

last_time = time.time()
running = True

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
photographer = PhotoGrapher(show=True)
messenger = Messenger(control_host, control_port)
print('Connected.')

while True:
    keys = key_check(['W', 'A', 'S', 'D'])
    if keys:
        key = keys[0]
        cmd = key_to_command[key]
        messenger.send_cmd(cmd)

    image = photographer.capture()

    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if list(faces):
        x, y, w, h = faces[0]

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        x = x + w / 2
        y = y + h / 2

        x /= image.shape[1]
        y /= image.shape[0]

        keys = []
        if x <= 0.35:
            keys.append('A')
        elif x >= 0.65:
            keys.append('D')
        if y <= 0.35:
            keys.append('W')
        elif y >= 0.65:
            keys.append('S')

        if keys and time.time() - last_time > 0.3:
            key = keys[0]
            cmd = key_to_command[key]
            messenger.send_cmd(cmd)
            last_time = time.time()

    cv2.imshow('face', image)
    cv2.waitKey(1)
