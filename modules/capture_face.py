import cv2
import os
import json
import numpy as np

def setup_direktori():
    # Bikin path otomatis, ga peduli kau run dari mana
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'dataset')
    cascades_dir = os.path.join(base_dir, 'cascades')
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(cascades_dir, exist_ok=True)
    return base_dir, dataset_dir, cascades_dir

def urus_id_nama(base_dir, nama_user):
    # Logika anti-hardcode pake JSON
    json_path = os.path.join(base_dir, 'names.json')
    names_map = {}
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                names_map = json.load(f)
        except Exception:
            pass # Kalau JSON rusak, anggap kosong
            
    # Cek kalau orangnya udah pernah didaftarin
    for id_key, name_val in names_map.items():
        if name_val.lower() == nama_user.lower():
            return id_key
            
    # Kalau orang baru, bikin ID baru
    new_id = str(len(names_map) + 1)
    names_map[new_id] = nama_user
    with open(json_path, 'w') as f:
        json.dump(names_map, f, indent=4)
        
    return new_id

def cek_kualitas_gambar(gray_roi):
    # Fitur Ultra: Cek apakah gambar ngeblur atau kegelapan
    blur_score = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
    brightness = np.mean(gray_roi)
    return blur_score > 15 and brightness > 40 # Angka aman biar ga nyimpen ampas

def capture_ultra():
    base_dir, dataset_dir, cascades_dir = setup_direktori()
    cascade_path = os.path.join(cascades_dir, 'haarcascade_frontalface_default.xml')
    
    if not os.path.exists(cascade_path):
        print(f"[ERROR] Mata kau buta? File {cascade_path} ga ada. Taruh dlu sana!")
        return

    detector = cv2.CascadeClassifier(cascade_path)
    
    nama_user = input("[?] Masukkan Nama Target: ").strip()
    if not nama_user:
        print("[ERROR] Nginput nama aja ga becus. Ulang!")
        return

    user_id = urus_id_nama(base_dir, nama_user)
    
    # Bikin folder spesifik misal: dataset/1_michael
    folder_name = f"{user_id}_{nama_user.replace(' ', '_')}"
    user_dir = os.path.join(dataset_dir, folder_name)
    os.makedirs(user_dir, exist_ok=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Kamera kau ga detect. Benerin dlu.")
        return

    cam.set(3, 640)
    cam.set(4, 480)
    
    count = 0
    max_pics = 100 # Kita gas 100 foto biar badag beneran
    print(f"\n[*] Target: {nama_user.upper()} | ID: {user_id}")
    print("[*] Liat kamera. Jgn banyak gerak. Jgn ada org lain di blkg kau!")

    while True:
        ret, img = cam.read()
        if not ret: continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # minSize biar dia ga nangkep muka semut di kejauhan
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100))

        # Fitur Ultra: Nolak kalau ada lebih dari 1 muka
        if len(faces) > 1:
            cv2.putText(img, "WARNING: ADA MUKA LAIN!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif len(faces) == 1:
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Fitur Ultra: Nolak kalau ngeblur/gelap
            if cek_kualitas_gambar(face_roi):
                count += 1
                file_path = os.path.join(user_dir, f"{count}.jpg")
                cv2.imwrite(file_path, face_roi)
                
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"Data: {count}/{max_pics}", (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 1)
            else:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(img, "GELAP/BLUR!", (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)

        cv2.imshow('Capture Wajah Ultra (ESC buat batal)', img)

        k = cv2.waitKey(50) & 0xff
        if k == 27 or count >= max_pics:
            break

    print(f"\n[OK] {count} foto berkualitas tinggi tersimpan di {user_dir}.")
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_ultra()