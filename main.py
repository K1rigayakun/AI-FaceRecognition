import cv2
import os
import json

# Import fungsi dari file modul yang udah kita bikin
# Pastikan nama fungsinya sesuai sama yang di file sebelumnya
from modules.capture_face import capture_ultra
from modules.train_model import train_badag

def deteksi_wajah():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cascade_path = os.path.join(base_dir, 'cascades', 'haarcascade_frontalface_default.xml')
    model_path = os.path.join(base_dir, 'trainer', 'trainer.yml')
    json_path = os.path.join(base_dir, 'names.json')

    # Fitur Ultra: Cek dulu modelnya ada ga. Kalau ga ada, ngapain deteksi?
    if not os.path.exists(model_path):
        print("[ERROR] Model AI kau belum ada. Train data dulu, jangan asal buka deteksi!")
        return

    # Load otak AI-nya
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(model_path)
    except AttributeError:
        print("[ERROR] Modul LBPH ga ada. Pastikan install opencv-contrib-python, bukan yg biasa!")
        return

    detector = cv2.CascadeClassifier(cascade_path)

    # Load daftar nama (Kamus ID -> Nama)
    names_map = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            names_map = json.load(f)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Webcam kau ga jalan. Benerin dulu kabelnya.")
        return

    cam.set(3, 640) # Lebar
    cam.set(4, 480) # Tinggi

    print("\n[*] Memulai sistem keamanan ultra... Tekan ESC untuk keluar.")

    while True:
        ret, img = cam.read()
        if not ret: break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Tebak ini muka siapa
            id_prediksi, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            # Info buat kau: Confidence di LBPH itu ngitung 'jarak' keakuratan.
            # Makin KECIL angkanya, makin akurat/mirip. Kalau 0 berarti sempurna.
            # Kita set batas toleransi di 75. Lewat dari itu, anggap orang asing.
            if confidence < 75:
                # Ambil nama dari JSON pake string ID. Kalau ga ketemu, tulis Unknown.
                nama = names_map.get(str(id_prediksi), "Unknown")
                akurasi = f" {round(100 - confidence)}%"
                warna = (0, 255, 0) # Hijau kalau kenal
            else:
                nama = "Unknown"
                akurasi = f" {round(100 - confidence)}%"
                warna = (0, 0, 255) # Merah kalau ga kenal

            # Gambar kotak dan teks di layar
            cv2.rectangle(img, (x, y), (x+w, y+h), warna, 2)
            cv2.putText(img, str(nama), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, warna, 2)
            cv2.putText(img, str(akurasi), (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.imshow('Face Recognition Ultra', img)
        
        # Keluar kalau pencet ESC
        if cv2.waitKey(10) & 0xff == 27:
            break

    print("[*] Kamera dimatikan.")
    cam.release()
    cv2.destroyAllWindows()

def main_menu():
    while True:
        print("\n" + "="*40)
        print("  SISTEM FACE RECOGNITION ANTI HARDCODE  ")
        print("="*40)
        print("1. Registrasi Wajah Baru (Capture)")
        print("2. Proses Pembelajaran (Train Model)")
        print("3. Mulai Deteksi Wajah (Live)")
        print("4. Keluar")
        print("="*40)
        
        pilihan = input(">> Pilih menu (1/2/3/4): ").strip()

        if pilihan == '1':
            capture_ultra()
        elif pilihan == '2':
            train_badag()
        elif pilihan == '3':
            deteksi_wajah()
        elif pilihan == '4':
            print("[*] Program dimatikan. Bye.")
            break
        else:
            print("[ERROR] Matamu, menunya cuma ada 1-4. Pilih yang bener!")

if __name__ == "__main__":
    main_menu()