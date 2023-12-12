import cv2, os, numpy as np
import random
import logging
import time
from paho.mqtt import client as mqtt_client

wajahDir = 'datawajah'
latihDir = 'latihwajah/training5.xml'
broker = 'cyb-iot.cloud.shiftr.io'
port = 1883
topic = "facerecognition/python/1"
topic2 = "facerecognition/python/2"
topic3 = "facerecognition/python/3"
topic4 = "facerecognition/python/train"
client_id = f'python-mqtt-{random.randint(0, 1000)}'
username = 'cyb-iot'
password = '9ptdvVF3Cuzouwiv'


def connect_mqtt():
    client = mqtt_client.Client(client_id)
    client.username_pw_set(username, password)
    client.connect(broker, port)
    return client


def on_disconnect(client, userdata, rc):
    logging.info("Disconnected with result code: %s", rc)


def publish(client, msg_count):
    msg = f"{msg_count} Hadir"
    result = client.publish(topic, msg)
    status = result[0]
    if status == 0:
        print(f"Send `{msg}` to topic `{topic}`")
    else:
        logging.error("Failed to send message to topic %s", topic)

def publish2(client, msg_count):
    msg = f"{msg_count} Hadir"
    result = client.publish(topic2, msg)
    status = result[0]
    if status == 0:
        print(f"Send `{msg}` to topic `{topic2}`")
    else:
        logging.error("Failed to send message to topic %s", topic2)

def publish3(client, msg_count):
    msg = f"{msg_count} Hadir"
    result = client.publish(topic3, msg)
    status = result[0]
    if status == 0:
        print(f"Send `{msg}` to topic `{topic3}`")
    else:
        logging.error("Failed to send message to topic %s", topic3)

def publish4(client, file_path):
    with open(file_path, 'r') as file:
        msg = file.read()
    result = client.publish(topic4, msg)
    status = result[0]
    if status == 0:
        print(f"Send `{msg}` to topic `{topic4}`")
    else:
        logging.error("Failed to send message to topic %s", topic4)

def main():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 1280)
    cam.set(4, 720)
    faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceRecognizer.read(latihDir)
    font = cv2.FONT_HERSHEY_COMPLEX
    id = 0
    names = ['tidakdiketahui', 'Ghopin', 'Ripki', 'widi']

    minWidth = 0.1 * cam.get(3)
    minHeight = 0.1 * cam.get(4)

    client = connect_mqtt()
    client.on_disconnect = on_disconnect

    frame_count = 0
    start_time = time.time()

    while True:
        retV, frame = cam.read()
        abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(abuAbu, 1.3, 5, minSize=(round(minWidth), round(minHeight)))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 0), 2)
            id, confidence = faceRecognizer.predict(abuAbu[y:y + h, x:x + w])
            persen = round(100 - confidence)
            confidencetext = "{0}%".format(round(persen))

            if persen > 60:
                nameID = names[id]
            else:
                nameID = names[0]
            print(nameID + '==' + confidencetext)
            print((x + w) / 2)

            frame_count += 1
            if frame_count >= 9:
                end_time = time.time()
                elapsed_time = end_time - start_time
                fps = round(frame_count / elapsed_time, 2)
                cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                frame_count = 0
                start_time = time.time()

            cv2.putText(frame, str(nameID), (x + 5, y - 5), font, 1, (0, 225, 0), 2)
            cv2.putText(frame, str(confidencetext), (x + 5, y + h - 5), font, 1, (0, 225, 0), 1)
            # client.loop_start()
            publish4(client, latihDir)
            if id == 1:
                publish(client, names[1])
            elif id == 2:
                publish2(client, names[2])
            elif id == 3:
                publish(client, names[3])
            else:
                print('notyet')
            # client.loop_stop()

        cv2.imshow('Webcam guwe', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('g'):
            break

    print('EXIT')
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
