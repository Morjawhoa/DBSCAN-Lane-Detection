import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import socket
import keyboard
import threading
import time


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


def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (width, height // 2),
        (0, height // 2),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, maxLineGap=50)
    return lines


def draw_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    combined_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return combined_image


def decide_direction(lines, image_width):
    if lines is None:
        return "STOP"

    left_lane = []
    right_lane = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)
        if slope < 0:
            left_lane.append(line)
        else:
            right_lane.append(line)

    left_xs = [x for line in left_lane for x in (line[0][0], line[0][2])]
    right_xs = [x for line in right_lane for x in (line[0][0], line[0][2])]

    if not left_xs or not right_xs:
        return "FORWARD"

    left_x_avg = np.mean(left_xs)
    right_x_avg = np.mean(right_xs)
    lane_center = (left_x_avg + right_x_avg) / 2
    frame_center = image_width / 2

    if lane_center < frame_center - 20:
        return "LEFT"
    elif lane_center > frame_center + 20:
        return "RIGHT"
    else:
        return "FORWARD"


if __name__ == '__main__':
    path_image = '../images/e89772fdd96947b1b7c2e603232a3f21.png'

    image = cv2.imread(path_image)
    lines = process_image(image)
    image = draw_lines(image, lines)

    cv2.imshow('image', image)
    cv2.waitKey(0)

