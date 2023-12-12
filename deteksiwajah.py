import cv2, os
wajahDir ='datawajah'
extension = 'jpg'
faceID = input('masukan face ID [lalu tekan enter] : ')
directory = 'datawajah/'+str(faceID)

# def hapus_file_gambar(directory, extension):
#     for filename in os.listdir(directory):
#         if filename.endswith(extension):
#             file_path = os.path.join(directory, filename)
#             os.remove(file_path)
#             print(f"File {filename} berhasil dihapus.")
def buatfolder():
    setDir = os.path.join(wajahDir,faceID)
    try:
        os.makedirs(setDir)
        print('menyiapkan direktori')
    except FileExistsError:
        print("Direktori sudah ada.")
        for filename in os.listdir(directory):
            if filename.endswith(extension):
                file_path = os.path.join(directory, filename)
                os.remove(file_path)
                print(f"File {filename} berhasil diganti.")

        # hapus_file_gambar('datawajah/'+str(faceID), 'jpg')


buatfolder()

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 1280)
cam.set(4, 720)
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyedetector = cv2.CascadeClassifier('haarcascade_eye.xml')
print('Tatap webcam dan tunggu prosesnya...')
ambildata = 1

while True:
    retV, frame = cam.read()
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuAbu, 1.3, 5)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 225, 225), 2)
        namafile = 'pict'+str(faceID)+'.'+str(ambildata)+'.jpg'
        roiAbuAbu = abuAbu[y:y+h,x:x+w]
        roiWarna = frame[y:y+h,x:x+w]
        cv2.imwrite(wajahDir + '/' + str(faceID) + '/' + namafile, roiAbuAbu)
        ambildata += 1
        eyes = eyedetector.detectMultiScale(roiAbuAbu, 1.3, 5)
        for (xe,ye,we,he) in eyes:
            cv2.rectangle(roiWarna,(xe,ye),(xe+we,ye+he),(0,0,225),1)

    cv2.imshow('Webcam guwe', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('g'):
        break
    elif ambildata>=100:
        break
print('pengambilan data selesai')
cam.release()
cv2.destroyAllWindows()