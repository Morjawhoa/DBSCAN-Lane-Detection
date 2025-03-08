import requests
from PIL import Image
from io import BytesIO
import socket
import cv2
import numpy as np
import keyboard
import time
import threading
import queue
from detect_keyboard_keys import key_check


key_to_command = {'W': 'ONA',
                  'S': 'ONB',
                  'A': 'ONC',
                  'D': 'OND',
                  }

direction_to_command = {
                        "FORWARD": "ONA",
                        "BACKWARD": "ONB",
                        "FORWARD_LEFT": "ONC",
                        "FORWARD_RIGHT": "OND",
                        "LEFT": "ONG",
                        "RIGHT": "ONH",
                        "STOP": "ONE"
                        }


def get_camera_image(url):
    response = requests.get(url)
    img_array = np.array(Image.open(BytesIO(response.content)))
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


class PhotoGrapher:
    def __init__(self, show=False):
        self.queue = queue.Queue()
        self.flag = True
        self.show = show

        self.refresh_thread = threading.Thread(target=self.refresh)
        self.refresh_thread.start()

    def refresh(self):
        while self.flag:
            image = get_camera_image(camera_url)

            if self.show:
                cv2.imshow('Camera', image)

            if self.queue.qsize() <= 3:
                self.queue.put(image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.flag = False

    def capture(self):
        image = self.queue.get()
        return image


class Messenger:
    def __init__(self, host, port, cd=0.3, enable=True):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((host, port))

        self.time = time.time()
        self.flag = True
        self.enable = enable
        self.cd = cd

    def send_cmd(self, command):
        if time.time() - self.time >= self.cd:
            if not self.enable:
                # print('Stop sending:', command)
                return 'disable'
            else:
                self.s.sendall(command.encode())
                # print('Sending:', command)
                self.time = time.time()
                return command
        else:
            return 'cd'

    def send_dir(self, direction):
        command = direction_to_command[direction]
        response = self.send_cmd(command)
        return response


camera_url = 'http://192.168.1.1:8080/?action=snapshot'
control_host = '192.168.1.1'
control_port = 2001


if __name__ == '__main__':
    last_time = time.time()
    running = True

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

        cv2.waitKey(1)





