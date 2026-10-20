**Instal OpenCV versi 4.13**
python -m pip install opencv-python==4.13.0.92
Versi ini tersedia di PyPI dan mendukung penggunaan CascadeClassifier

**Pastikan instalasi berhasil**
python -c "import cv2; print('Versi:', cv2.__version__); print('CascadeClassifier:', hasattr(cv2, 'CascadeClassifier')); print('Haar:', cv2.data.haarcascades)"
Hasil yang diharapkan:
Versi: 4.13.0
CascadeClassifier: True
Haar: ...\site-packages\cv2\data\

**Unduh model AI untuk deteksi umur**
Di terminal VS Code, pastikan Anda berada di folder proyek:
_</>powershell_
cd C:\wamp64\www\IoT_python_camera

Buat folder untuk menyimpan model:
_</>powershell_
New-Item -ItemType Directory -Force models

Unduh dua file berikut menggunakan PowerShell:
_</>powershell_
curl.exe -L "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt" -o "models\age_deploy.prototxt"
curl.exe -L "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel" -o "models\age_net.caffemodel"

Setelah selesai, periksa file-nya:
_</>powershell_
Get-ChildItem .\models

Seharusnya ada dua file:
models
├── age_deploy.prototxt
└── age_net.caffemodel
