import os
import base64
import time
import json
from datetime import datetime
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Flask uygulamasını başlat
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Global değişkenler ve model yükleme
HAS_YOLO = False
yolo_model = None

try:
    from ultralytics import YOLO
    # YOLOv8n (nano) modelini yükle, bulutta hızlı çalışır
    # Colab'de GPU varsa otomatik kullanır
    yolo_model = YOLO('yolov8n.pt')
    HAS_YOLO = True
    print("[INFO] YOLOv8 modeli başarıyla yüklendi.")
except Exception as e:
    print(f"[WARNING] YOLOv8 yüklenemedi (ultralytics bulunamadı veya hata oluştu): {e}")
    print("[INFO] Sistem OpenCV Haar Cascade (Yüz Algılama) yedek moduna geçiyor.")

# Haar Cascade Yedek Yüz Algılayıcı (YOLO yüklü değilse veya yedek olarak kullanılacak)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mediapipe El Algılayıcı
HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    HAS_MEDIAPIPE = True
    print("[INFO] Mediapipe Hands başarıyla yüklendi.")
except Exception as e:
    print(f"[WARNING] Mediapipe Hands yüklenemedi: {e}")

# İhlal logları ve istatistikleri (Bellekte tutulur)
event_logs = []
cumulative_counts = {}

def apply_opencv_filter(image, filter_type):
    """
    Kullanıcının seçtiği OpenCV görüntü işleme filtresini uygular.
    """
    if filter_type == 'grayscale':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif filter_type == 'canny':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Gürültü azaltma ve kenar algılama
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blurred, 50, 150)
        return cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    elif filter_type == 'blur':
        return cv2.GaussianBlur(image, (15, 15), 0)
    elif filter_type == 'threshold':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return image

@app.route('/')
def index():
    """Ana sayfa yönlendirmesi"""
    return render_template('index.html', has_yolo=HAS_YOLO)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    t_start = time.time()
    
    # JSON verisini al
    data = request.get_json(silent=True) or {}
    
    # Base64 görüntüyü al ve çöz
    img_data_b64 = data.get('image', '')
    if not img_data_b64:
        return jsonify({'error': 'Görüntü verisi bulunamadı'}), 400
        
    if ',' in img_data_b64:
        img_data_b64 = img_data_b64.split(',')[1]
        
    try:
        img_bytes = base64.b64decode(img_data_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': f'Görüntü çözülemedi: {str(e)}'}), 400
        
    if frame is None:
        return jsonify({'error': 'Görüntü çözülemedi (Boş kare)'}), 400

    # Ayna modu kontrolü (Görüntüyü Yatay Çevir)
    mirror = data.get('mirror', False)
    if mirror:
        frame = cv2.flip(frame, 1)

    # Parametreleri al
    polygon_ratios = data.get('polygon', [])
    filter_type = data.get('filter', 'none')
    conf_threshold = float(data.get('confidence', 0.25))
    target_classes = data.get('target_classes', ['person', 'car', 'dog', 'cat', 'bicycle', 'hand'])

    height, width = frame.shape[:2]
    
    # Çokgen (Yasaklı Bölge) piksel koordinatlarını hesapla
    polygon_pts = []
    if polygon_ratios:
        polygon_pts = np.array([[int(pt[0] * width), int(pt[1] * height)] for pt in polygon_ratios], dtype=np.int32)
        # OpenCV işlemleri için format
        if len(polygon_pts) > 0:
            polygon_pts = polygon_pts.reshape((-1, 1, 2))

    # Yasaklı bölge maskesi oluştur (En ufak temas/kesişimi tespit etmek için)
    polygon_mask = None
    if len(polygon_pts) >= 3:
        polygon_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [polygon_pts], 255)

    detections = []
    intrusion_detected = False
    class_counts = {}

    # 1. YOLOv8 ile Tahmin ve Takip Aşaması
    if HAS_YOLO:
        results = yolo_model(frame, conf=conf_threshold, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Sınıf adını al
                cls_name = yolo_model.names[cls_id]
                
                # Eğer sınıf hedef listemizde yoksa atla
                if cls_name not in target_classes:
                    continue
                
                # Bounding box koordinatları [x1, y1, x2, y2]
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Kesişim maskesi ile hassas ihlal denetimi (Gereksiz boş alanların ihlal vermesini önlemek için daraltılmış kutu)
                is_inside = False
                if polygon_mask is not None:
                    w = x2 - x1
                    h = y2 - y1
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    if cls_name == 'person':
                        # İnsan için genişliği %40'a daralt (kolları ve boşlukları ele, gövdeyi kontrol et)
                        sx1 = max(0, int(cx - 0.20 * w))
                        sx2 = min(width - 1, int(cx + 0.20 * w))
                        sy1 = y1
                        sy2 = y2
                    else:
                        # Diğer nesneler için genel kutuyu %70'e daralt
                        sx1 = max(0, int(cx - 0.35 * w))
                        sx2 = min(width - 1, int(cx + 0.35 * w))
                        sy1 = max(0, int(cy - 0.35 * h))
                        sy2 = min(height - 1, int(cy + 0.35 * h))
                    
                    box_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.rectangle(box_mask, (sx1, sy1), (sx2, sy2), 255, -1)
                    overlap = cv2.bitwise_and(polygon_mask, box_mask)
                    if np.any(overlap > 0):
                        is_inside = True
                        intrusion_detected = True
                
                px, py = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Sınıf sayısını arttır
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                
                detections.append({
                    'class': 'person' if cls_name == 'person' else cls_name,
                    'confidence': round(conf, 2),
                    'box': [x1, y1, x2, y2],
                    'is_intruder': is_inside,
                    'center': [px, py]
                })

    # 2. Mediapipe ile El Algılama (YOLOv8'de el sınıfı olmadığı için paralel çalışır)
    if HAS_MEDIAPIPE and 'hand' in target_classes:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = hands_detector.process(rgb_frame)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                // Elin sınır koordinatlarını (bounding box) hesapla
                x_max = 0.0
                y_max = 0.0
                x_min = 1.0
                y_min = 1.0
                for lm in hand_landmarks.landmark:
                    x_min = min(x_min, lm.x)
                    y_min = min(y_min, lm.y)
                    x_max = max(x_max, lm.x)
                    y_max = max(y_max, lm.y)
                
                # Normalize koordinatları piksel koordinatlarına dönüştür
                x1 = max(0, int(x_min * width))
                y1 = max(0, int(y_min * height))
                x2 = min(width - 1, int(x_max * width))
                y2 = min(height - 1, int(y_max * height))
                
                # El kutusu ile yasaklı bölge kesişim denetimi (Hafifçe daraltılmış %85 boyut)
                is_inside = False
                if polygon_mask is not None:
                    w = x2 - x1
                    h = y2 - y1
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    sx1 = max(0, int(cx - 0.425 * w))
                    sx2 = min(width - 1, int(cx + 0.425 * w))
                    sy1 = max(0, int(cy - 0.425 * h))
                    sy2 = min(height - 1, int(cy + 0.425 * h))
                    
                    box_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.rectangle(box_mask, (sx1, sy1), (sx2, sy2), 255, -1)
                    overlap = cv2.bitwise_and(polygon_mask, box_mask)
                    if np.any(overlap > 0):
                        is_inside = True
                        intrusion_detected = True
                
                px, py = (x1 + x2) // 2, (y1 + y2) // 2
                cls_name = 'hand'
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                
                detections.append({
                    'class': 'El (Hand)',
                    'confidence': 0.90, # Mediapipe el için sabit yüksek güven
                    'box': [x1, y1, x2, y2],
                    'is_intruder': is_inside,
                    'center': [px, py]
                })

    # 3. Yüz Algılama (OpenCV Haar Cascade) - Yakın çekimde YOLOv8 insanı kaçırırsa yüz üzerinden yakalamak için paralel çalışır
    if 'person' in target_classes:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # Eğer YOLO zaten bu yüzü kapsayan bir insan algıladıysa mükerrer algılamayı önleyelim
            is_duplicate = False
            for det in detections:
                if det['class'] == 'person':
                    px1, py1, px2, py2 = det['box']
                    if px1 <= x + w//2 <= px2 and py1 <= y + h//2 <= py2:
                        is_duplicate = True
                        break
            
            if is_duplicate:
                continue
                
            # Yüz kutusu ile yasaklı bölge kesişim denetimi (Hafifçe daraltılmış %80 boyut)
            is_inside = False
            if polygon_mask is not None:
                w_face = x2 - x1
                h_face = y2 - y1
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                sx1 = max(0, int(cx - 0.40 * w_face))
                sx2 = min(width - 1, int(cx + 0.40 * w_face))
                sy1 = max(0, int(cy - 0.40 * h_face))
                sy2 = min(height - 1, int(cy + 0.40 * h_face))
                
                box_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.rectangle(box_mask, (sx1, sy1), (sx2, sy2), 255, -1)
                overlap = cv2.bitwise_and(polygon_mask, box_mask)
                if np.any(overlap > 0):
                    is_inside = True
                    intrusion_detected = True
            
            px, py = (x1 + x2) // 2, (y1 + y2) // 2
            cls_name = 'person'
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            
            detections.append({
                'class': 'person', # Kutucuğun hemen üstünde person yazması için
                'confidence': 0.85,
                'box': [x1, y1, x2, y2],
                'is_intruder': is_inside,
                'center': [px, py]
            })

    # 4. Görüntü İşleme Filtresini Uygula (Görsel Katman)
    processed_frame = apply_opencv_filter(frame.copy(), filter_type)

    # 5. Görsel Overlay ve Çizimler
    zone_color = (0, 0, 255) if intrusion_detected else (0, 255, 0)
    zone_thickness = 3 if intrusion_detected else 2
    
    if len(polygon_pts) > 0:
        cv2.polylines(processed_frame, [polygon_pts], isClosed=True, color=zone_color, thickness=zone_thickness)
        overlay = processed_frame.copy()
        cv2.fillPoly(overlay, [polygon_pts], zone_color)
        alpha = 0.15 if not intrusion_detected else 0.3
        cv2.addWeighted(overlay, alpha, processed_frame, 1 - alpha, 0, processed_frame)

    # Algılanan nesneleri çiz
    for det in detections:
        x1, y1, x2, y2 = det['box']
        is_intruder = det['is_intruder']
        cls_name = det['class']
        conf = det['confidence']
        
        box_color = (0, 0, 255) if is_intruder else (255, 0, 128)
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), box_color, 2)
        
        label = f"{cls_name} %{int(conf*100)}"
        if is_intruder:
            label += " [IHLAL!]"
            
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(processed_frame, (x1, y1 - 20), (x1 + w_text, y1), box_color, -1)
        cv2.putText(processed_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(processed_frame, tuple(det['center']), 4, (0, 255, 255), -1)

    # İhlal Durumu Banner'ı
    if intrusion_detected:
        cv2.rectangle(processed_frame, (0, 0), (width, 40), (0, 0, 255), -1)
        cv2.putText(processed_frame, "!!! YASAKLI BOLGE IHLALI !!!", (width // 2 - 150, 26), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        from datetime import timezone, timedelta
        tr_tz = timezone(timedelta(hours=3))
        now_str = datetime.now(tr_tz).strftime("%H:%M:%S")
        msg = f"Yasaklı Bölge İhlali Tespit Edildi!"
        
        if not event_logs or (time.time() - event_logs[-1]['time_raw'] > 1.5):
            event_logs.append({
                'timestamp': now_str,
                'message': msg,
                'type': 'danger',
                'time_raw': time.time()
            })
            cumulative_counts['violations'] = cumulative_counts.get('violations', 0) + 1

    # Bilgi Banner'ı
    status_text = "YOLOv8 Aktif" if HAS_YOLO else "OpenCV Yedek Mod (Yüz Algilama)"
    status_color = (0, 255, 0) if HAS_YOLO else (0, 165, 255)
    cv2.putText(processed_frame, status_text, (10, height - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)

    # FPS hesaplama
    fps = round(1.0 / (time.time() - t_start), 1)

    # Görüntüyü Base64'e dönüştür
    _, buffer = cv2.imencode('.jpg', processed_frame)
    processed_b64 = base64.b64encode(buffer).decode('utf-8')
    processed_b64_src = f"data:image/jpeg;base64,{processed_b64}"

    if len(event_logs) > 15:
        event_logs.pop(0)

    return jsonify({
        'image': processed_b64_src,
        'intrusion_detected': intrusion_detected,
        'detections': [{'class': d['class'], 'confidence': d['confidence'], 'is_intruder': d['is_intruder']} for d in detections],
        'counts': class_counts,
        'cumulative_violations': cumulative_counts.get('violations', 0),
        'fps': fps,
        'logs': [{'timestamp': log['timestamp'], 'message': log['message'], 'type': log['type']} for log in reversed(event_logs)]
    })

@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    global event_logs, cumulative_counts
    event_logs = []
    cumulative_counts = {'violations': 0}
    return jsonify({'status': 'success', 'message': 'İstatistikler sıfırlandı.'})

if __name__ == '__main__':
    try:
        from waitress import serve
        print("[INFO] Üretim WSGI Sunucusu (Waitress) başlatılıyor... Port: 5000")
        serve(app, host='0.0.0.0', port=5000)
    except ImportError:
        print("[WARNING] Waitress modülü yüklenemedi. Flask geliştirme sunucusu kullanılıyor...")
        app.run(host='0.0.0.0', port=5000, debug=False)
