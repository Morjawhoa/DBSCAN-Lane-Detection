import time
from Controller import PhotoGrapher, Messenger, key_to_command
import cv2
from DBSCAN import detect_DBSCAN
from tqdm import tqdm
import signal
import sys
import keyboard


def signal_handler(sig, frame):
    out.release()
    cv2.destroyAllWindows()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

camera_url = 'http://192.168.1.1:8080/?action=snapshot'
control_host = '192.168.1.1'
control_port = 2001

episode_length = 1000
cd = 0.15
running = True
mode = 'track'

photographer = PhotoGrapher(show=False)
messenger = Messenger(control_host, control_port, cd=cd, enable=True)
print('Connected.')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 5.0, (320, 240))  # Adjust resolution if necessary

while True:
    while mode == 'lane':
        pbar = tqdm(total=episode_length)
        for i in range(episode_length):
            image = photographer.capture()
            image_annotated, direction = detect_DBSCAN(image)

            response = messenger.send_dir(direction)

            cv2.imshow('Lane Detection', image_annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break

            out.write(image_annotated)

            pbar.update(1)

            pbar.set_postfix({'send': response,
                              'direction': direction})

            if keyboard.is_pressed('left'):
                mode = 'lane'
                print('lane')
                time.sleep(0.5)
                response = messenger.send_cmd('ON2')
            elif keyboard.is_pressed('right'):
                mode = 'track'
                print('track')
                time.sleep(0.5)
                response = messenger.send_cmd('ON1')
                break
        del pbar

    while mode == 'track':
        if keyboard.is_pressed('left'):
            mode = 'lane'
            print('lane')
            time.sleep(0.5)
            response = messenger.send_cmd('ON2')
            break
        elif keyboard.is_pressed('right'):
            mode = 'track'
            print('track')
            time.sleep(0.5)
            response = messenger.send_cmd('ON1')
        time.sleep(0.01)


out.release()
cv2.destroyAllWindows()
