
import cv2
from pathlib import Path

# ==========================================
# 1. MENENTUKAN LOKASI FILE MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

AGE_CONFIG = BASE_DIR / "models" / "age_deploy.prototxt"
AGE_WEIGHTS = BASE_DIR / "models" / "age_net.caffemodel"

# Rentang usia yang dapat diprediksi model
AGE_RANGES = [
    "0-2 tahun",
    "4-6 tahun",
    "8-12 tahun",
    "15-20 tahun",
    "25-32 tahun",
    "38-43 tahun",
    "48-53 tahun",
    "60-100 tahun"
]

# Nilai rata-rata warna yang dibutuhkan model
MODEL_MEAN_VALUES = (
    78.4263377603,
    87.7689143744,
    114.895847746
)

# Pastikan kedua file model tersedia
if not AGE_CONFIG.is_file() or not AGE_WEIGHTS.is_file():
    print("File model umur tidak ditemukan.")
    print("Periksa folder models.")
    raise SystemExit(1)

# ==========================================
# 2. MEMUAT MODEL DETEKSI
# ==========================================

face_model = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

eye_model = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_eye.xml"
)

# Memuat model AI untuk memperkirakan umur
try:
    age_model = cv2.dnn.readNetFromCaffe(
        str(AGE_CONFIG),
        str(AGE_WEIGHTS)
    )
except cv2.error as error:
    print("Gagal memuat model umur:", error)
    raise SystemExit(1)

if face_model.empty() or eye_model.empty():
    print("Model deteksi wajah atau mata gagal dimuat.")
    raise SystemExit(1)

print("Model wajah, mata, dan umur berhasil dimuat.")

# ==========================================
# 3. MEMBUKA KAMERA LAPTOP
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Tidak dapat membuka kamera.")
    raise SystemExit(1)

print("Deteksi wajah, mata, dan umur aktif.")
print("Tekan Q untuk keluar.")

# ==========================================
# 4. MEMBACA VIDEO
# ==========================================

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Gagal membaca gambar dari kamera.")
            break

        # Mengubah gambar menjadi grayscale
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Mendeteksi wajah
        faces = face_model.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        total_eyes = 0

        # ==========================================
        # 5. MEMPROSES SETIAP WAJAH
        # ==========================================

        for (x, y, w, h) in faces:

            # Menggambar kotak hijau pada wajah
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Mengambil gambar wajah berwarna
            face_image = frame[y:y + h, x:x + w]

            # Menyiapkan gambar untuk model umur
            blob = cv2.dnn.blobFromImage(
                face_image,
                1.0,
                (227, 227),
                MODEL_MEAN_VALUES,
                swapRB=False
            )

            # Meminta model memprediksi rentang umur
            age_model.setInput(blob)
            predictions = age_model.forward()

            # Mengambil kategori dengan nilai prediksi tertinggi
            age_index = int(predictions[0].argmax())
            age_label = AGE_RANGES[age_index]

            # Menampilkan perkiraan umur di atas wajah
            label_y = max(y - 10, 20)

            cv2.putText(
                frame,
                f"Perkiraan umur: {age_label}",
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            # ==========================================
            # 6. MENDETEKSI MATA
            # ==========================================

            face_gray = gray[y:y + h, x:x + w]

            # Fokus pada bagian atas wajah
            upper_height = int(h * 0.65)
            eye_area = face_gray[0:upper_height, :]

            eyes = eye_model.detectMultiScale(
                eye_area,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(15, 15)
            )

            total_eyes += len(eyes)

            for (ex, ey, ew, eh) in eyes:
                # Menggambar kotak biru di area mata
                cv2.rectangle(
                    frame,
                    (x + ex, y + ey),
                    (x + ex + ew, y + ey + eh),
                    (255, 0, 0),
                    2
                )

        # ==========================================
        # 7. MENAMPILKAN INFORMASI
        # ==========================================

        cv2.putText(
            frame,
            f"Jumlah wajah: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Area mata: {total_eyes}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        # Menampilkan hasil kamera
        cv2.imshow(
            "Smart Room - Deteksi Umur dan Mata",
            frame
        )

        # Keluar jika menekan Q atau q
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Program dihentikan.")
