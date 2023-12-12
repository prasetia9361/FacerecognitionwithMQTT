import cv2
import os
import numpy as np
# import random
# import logging
# import time
import paho.mqtt.client as mqtt

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
    else:
        print(f"Failed to connect, return code: {rc}")

def on_publish(client, userdata, mid):
    print("Message published successfully")

def send_file_mqtt(file_path, topic, broker_address="cyb-iot.cloud.shiftr.io", port=1883, username="cyb-iot", password="9ptdvVF3Cuzouwiv"):
    client = mqtt.Client()
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.on_publish = on_publish

    client.connect(broker_address, port)

    with open(file_path, 'r') as file:
        file_content = file.read()

    client.publish(topic, file_content)
    print("publish sukses")

    client.disconnect()

datasetDir = 'datawajah'
latihDir = 'latihwajah/trainingmqtt.xml'
topic = "facerecognition/python/train"
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def bacaGambar(direktori):
    gambarWajah = []
    labels = []
    # jumlahdatas = []

    for root, dirs, files in os.walk(direktori):
        for file in files:
            imagePath = os.path.join(root, file)

            # Mendapatkan ID wajah dari direktori
            faceID = int(os.path.split(root)[-1])
            # jumlahdata = int(os.path.split(imagePath)[-1])
            print(faceID)
            # Baca gambar dan konversi ke grayscale
            img = cv2.imread(imagePath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


            # Deteksi wajah menggunakan cascade classifier
            faces = faceDetector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                gambarWajah.append(roi_gray)
                labels.append(faceID)
                # jumlahdatas.append(jumlahdata)

    return gambarWajah, labels
print('Mesin sedang melakukan training! wait ')
gambarWajah, labels = bacaGambar(datasetDir)

# Melatih model LBPH
faceRecognizer.train(gambarWajah, np.array(labels))
faceRecognizer.write(latihDir)
send_file_mqtt(latihDir, topic)
print('sebanyak'+format (len(np.unique(labels)))+ ' data telah ditraining')
