import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import socket
import keyboard
import threading
import time
from DBSCAN import detect_DBSCAN, find_lane_points, process_image


class CarController:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = None
        self.connect()

    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
        except Exception as e:
            print(f"Error connecting to {self.host}:{self.port} - {e}")

    def send_command(self, command):
        try:
            if self.socket is None:
                self.connect()
            self.socket.sendall(command.encode())
            response = self.socket.recv(1024)
            print('Received:', response.decode('utf-8', errors='ignore'))
        except ConnectionResetError as e:
            print(f"Connection reset error: {e}")
            self.connect()  # Reconnect on error
        except socket.error as e:
            print(f"Socket error: {e}")

    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None


def get_camera_image(url):
    response = requests.get(url)
    img_array = np.array(Image.open(BytesIO(response.content)))
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)






# def decide_direction(left_points, right_points, image_width):
#     if left_points and right_points:
#         return "FORWARD"
#     elif left_points:
#         left_x = left_points[0][0]
#         if left_x < image_width * 0.25:
#             return "FORWARD"
#         elif left_x > image_width * 0.25:
#             return "RIGHT"
#         else:
#             return "FORWARD_RIGHT"
#     elif right_points:
#         right_x = right_points[0][0]
#         if right_x > image_width * 0.75:
#             return "FORWARD"
#         elif right_x < image_width * 0.75:
#             return "LEFT"
#         else:
#             return "FORWARD_LEFT"
#     else:
#         return "STOP"


def draw_lane_points(image, left_points, right_points):
    for point in left_points:
        cv2.circle(image, point, 10, (0, 0, 255), -1)
    for point in right_points:
        cv2.circle(image, point, 10, (0, 255, 0), -1)
    return image


if __name__ == '__main__':
    path_image = '../images/20240704112131~1.jpg'
    # camera_url = 'http://192.168.1.1:8080/?action=snapshot'

    image = cv2.imread(path_image)
    # image = get_camera_image(camera_url)

    binary = process_image(image)
    image_annotated, direction = detect_DBSCAN(image)
    # print(direction)

    cv2.imwrite('虚线.jpg', image_annotated)

    cv2.namedWindow("binary_image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("binary_image", 960, 540)
    cv2.imshow('binary_image', image_annotated)
    cv2.waitKey(0)
