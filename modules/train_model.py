import cv2
import os
import numpy as np
from PIL import Image

def train_badag():
    # 1. Setup direktori otomatis
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'dataset')
    trainer_dir = os.path.join(base_dir, 'trainer')
    
    # Bikin folder trainer kalau kau lupa bikin
    os.makedirs(trainer_dir, exist_ok=True)

    # 2. Panggil algoritma LBPH (Ini otak recognizer-nya)
    # Kalau error di sini, berarti kau belum install opencv-contrib-python!
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        print("[ERROR] Modul LBPH ga ketemu! Kau pasti install opencv-python biasa. Install yg contrib!")
        return

    faces = []
    ids = []

    print("[*] Proses training dimulai. Jangan di-close, tunggu aja...")

    # 3. Cek apakah ada data buat di-train
    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
        print("[ERROR] Folder dataset kosong melompong. Capture wajah dulu!")
        return

    # 4. Looping baca semua file di dalam sub-folder dataset
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith("jpg") or file.endswith("png"):
                path = os.path.join(root, file)
                
                # Ambil ID dari nama folder. Format kita kan: dataset/1_michael/1.jpg
                folder_name = os.path.basename(root)
                try:
                    # Ambil angka sebelum underscore pertama
                    id_user = int(folder_name.split('_')[0])
                except ValueError:
                    print(f"[!] Ada folder aneh nggak sesuai format: {folder_name}. Di-skip!")
                    continue
                    
                # 5. Pake PIL buat mastiin format gambar stabil dan convert ke Grayscale ('L')
                try:
                    PIL_img = Image.open(path).convert('L') 
                    img_numpy = np.array(PIL_img, 'uint8')
                except Exception as e:
                    print(f"[!] Gagal baca file {path}: {e}")
                    continue

                faces.append(img_numpy)
                ids.append(id_user)

    if len(faces) == 0:
        print("[ERROR] Ga ada gambar valid yang bisa di-train. Cek dataset kau.")
        return

    # 6. Eksekusi Training
    print(f"[*] Menemukan {len(faces)} sampel gambar dari {len(np.unique(ids))} orang. Sedang memproses otak AI...")
    recognizer.train(faces, np.array(ids))

    # 7. Simpan hasil belajar ke file .yml
    model_path = os.path.join(trainer_dir, 'trainer.yml')
    recognizer.write(model_path)
    
    print(f"[OK] Mantap. Model berhasil di-training dan disave di: {model_path}")

if __name__ == "__main__":
    train_badag()