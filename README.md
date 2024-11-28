# Real Time Drowsiness Detection

Aplikasi ini dibangun menggunakan bahasa __python__ dengan library __tensorflow__ untuk melatih model _machine learning_ dan __Tkinkter__ untuk membangun _user interface_. Untuk mendeteksi wajah pengguna, saya menggunakan YOLO. YOLO dibuat dengan memberikan label pada setiap citra wajah pada dataset. Data yang digunakan untuk melakukan pelatihan model machine learning dapat di download dari <a href="https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd"> Kaggle</a>. Aplikasi ini berjalan secara realtime. Kamera pada komputer akan menangkap wajah dan sistem akan langsung melakukan klasifikasi wajah. Jika wajah terdeteksi mengantuk, maka kotak pendeteksi akan berwarna merah. Sebaliknya, jika tidak mengantuk maka kotak pendeteksi akan berwarna hijau.

untuk menggunakan aplikasi ini, jalankan file python (main.py). Atau bisa menggunakan syntax berikut:
```
python main.py
```

Untuk mengubah file python ke dalam bentuk executable program, diperlukan library __pyinstaller__ yang dapat diinstal dengan menggunakan code berikut pada terminal:
```
pip install pyinstaller
```
untuk merubah file, gunakan code berikut pada terminal:
```
pyinstaller main.py --windowed --onefile --collect-all tensorflow
``` 
<br>



