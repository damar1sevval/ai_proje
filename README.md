# Akıllı Güvenlik, Nesne Takibi ve Yasak Bölge İhlal Tespit Sistemi

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/damar1sevval/ai_proje/blob/main/sevval_ai_proje_son.ipynb)

Bu proje, **RYZ2002 Yapay Zeka Uygulamaları Dersi Dönem Sonu Projesi** kapsamında geliştirilmiş, görüntü işleme filtreleri ve yapay zeka nesne tespiti teknolojilerini harmanlayan interaktif bir **Akıllı Güvenlik Arayüzü**dür.

---

## 🎬 Proje Özellikleri

1. **Yapay Zeka Nesne Tespiti (YOLOv8):** Görüntü akışındaki kişileri, arabaları, kedi ve köpekleri gerçek zamanlı olarak algılar ve takip eder.
2. **Yedek Yüz Algılama Sistemi (OpenCV Haar Cascade):** Eğer sistemde YOLOv8 yüklenemezse veya GPU bulunamazsa, sistem çökmek yerine OpenCV Yüz Algılama moduna otomatik geçiş yapar.
3. **El Takip Modülü (Mediapipe Hands):** YOLOv8 modelinin algılamadığı el hareketlerini hassas bir şekilde yakalamak amacıyla paralel olarak çalışır.
4. **Çokgen Yasak Bölge (Restricted Area):** Canlı kamera görüntüsü üzerinde tıklama yaparak serbestçe çokgen şeklinde yasaklı bölge sınırları tanımlanabilir.
5. **Dinamik Görüntü İşleme Filtreleri:** Arayüzden anlık olarak *Canny Kenar Algılama, Grayscale, Gaussian Blur (Bulanıklaştırma)* ve *Eşikleme (Threshold)* filtreleri uygulanabilir.
6. **Canlı İstatistikler & Grafik (Chart.js):** Aktif tespit sayılarını ve ihlal durumlarını grafiksel olarak canlı gösterir.
7. **Sesli Alarm:** İhlal tespiti anında Web Audio API aracılığıyla tarayıcı üzerinden yüksek frekanslı sesli alarm verir.

---

## 🚀 Çalıştırma Talimatları

Projeyi **Google Colab** üzerinde veya **Yerel Bilgisayarınızda (Local)** çalıştırabilirsiniz.

### Seçenek A: Google Colab Üzerinde Çalıştırma (Hızlı & Kolay)

Sistem bulut üzerinde hızlı çalışacak şekilde optimize edilmiştir.

> [!NOTE]
> GitHub bazen Jupyter Notebook (`.ipynb`) dosyalarını kendi arayüzünde görüntülerken yükleme hatası verebilir (bu GitHub sunucularının anlık yoğunluğundan kaynaklanan kronik bir durumdur). Dosyada hiçbir bozukluk yoktur. Notebook'u doğrudan açmak için en üstte bulunan **"Open in Colab"** butonuna tıklayabilirsiniz.

1. Sayfanın en üstünde yer alan **Open in Colab** butonuna tıklayarak veya `sevval_ai_proje_son.ipynb` dosyasını tarayıcınızda açtığınız **Google Colab** ortamına yükleyerek açın.
2. Yukarıdaki menüden **Çalışma Zamanı > Çalışma zamanı türünü değiştir** yolunu izleyerek **GPU** (örn. T4 GPU) donanımını seçin.
3. Notebook içindeki hücreleri yukarıdan aşağıya sırayla çalıştırın.
4. En son hücreyi çalıştırdığınızda çıktıda belirecek olan **mavi renkli proxy bağlantısına (Google Colab tünel linki)** tıklayın.
5. Açılan web sayfasında tarayıcınızın kamera iznini onaylayın, ekrana tıklayarak yasaklı bölgenizi çizin ve sistemi test edin!

---

### Seçenek B: Yerel Bilgisayarda (Local) Çalıştırma

Gerekli kaynak kodlar notebook içerisinden ayıklanmış olup, bağımsız bir Flask projesi yapısında sunulmuştur.

#### 1. Bağımlılıkların Kurulması
Öncelikle bilgisayarınızda **Python 3.8+** yüklü olmalıdır. Proje ana klasöründeyken terminalinizi (komut satırı) açın ve gerekli kütüphaneleri kurun:

```bash
pip install ultralytics flask flask-cors waitress opencv-python numpy mediapipe
```

*(Not: Windows üzerinde bazı sistemlerde C++ derleme araçları gerekebilir. Hata alırsanız Python sanal ortamı (venv) oluşturup kurmanız önerilir).*

#### 2. Uygulamanın Başlatılması
Kurulum tamamlandıktan sonra aşağıdaki komutla Flask sunucusunu yerel olarak başlatın:

```bash
python app.py
```

#### 3. Tarayıcıda Açma
Sunucu çalıştıktan sonra tarayıcınızı açın ve aşağıdaki adrese gidin:
```
http://localhost:5000
```
- Ekranda kamera iznini onaylayın.
- **Kamerayı Başlat** butonuna tıklayın.
- Görüntü üzerine tıklayarak yasaklı bölge sınırlarınızı belirleyin.

---

## 🛠️ Klasör Yapısı

```
ai_proje/
│
├── app.py                            # Flask Backend Sunucusu (Görüntü İşleme ve AI)
├── sevval_ai_proje_son.ipynb         # Google Colab Jupyter Notebook Çalışması
├── sevval_damar_proje_raporu_son.docx # Proje Raporu Dokümanı
├── .gitignore                        # Git dışı bırakma listesi
├── README.md                         # Çalıştırma talimatları ve açıklama
│
├── templates/
│   └── index.html                    # Web arayüzü HTML şablonu
│
└── static/
    ├── css/
    │   └── style.css                 # Arayüz tasarım ve stil dosyası
    └── js/
        └── main.js                   # Tarayıcı tarafı kamera ve websocket-like API akış yönetimi
```

---

## 🎨 Arayüz Kontrolleri Nasıl Kullanılır?

- **Bölge Çizimi:** Ekrana en az 3 nokta yerleştirerek yasaklı alanınızı (çokgen) kapatın. Bu alana giren herhangi bir tanımlı nesne ihlal alarmını tetikleyecektir.
- **Bölgeyi Temizle:** Çizdiğiniz çokgeni silerek yeni bir alan çizmenizi sağlar.
- **Sesli Alarm:** İhlal anında çalacak olan sesli alarmı kapatıp açabilirsiniz.
- **Güven Eşiği (Confidence):** YOLOv8 nesne tanıma hassasiyetini ayarlar.
- **Ayna Modu:** Kamera görüntünüzün yatay simetrisini alır.
- **İstatistikleri Sıfırla:** Toplam ihlal sayacını ve grafik geçmişini sıfırlar.
