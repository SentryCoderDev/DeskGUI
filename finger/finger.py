import cv2
from cvzone.HandTrackingModule import HandDetector
import json # JSON kütüphanesini içe aktar

# JSON haritasını yükle
map_file_path = r'c:\Users\emirh\OneDrive\Masaüstü\hand_combinations_map.json' # Ham dize olarak tanımla
try:
    with open(map_file_path, 'r') as f:
        hand_map = json.load(f)
except FileNotFoundError:
    print(f"Hata: Harita dosyası bulunamadı: {map_file_path}")
    exit()
except json.JSONDecodeError:
    print(f"Hata: Harita dosyası geçerli bir JSON değil: {map_file_path}")
    exit()

cap = cv2.VideoCapture(0)

# İki eli algılamak için maxHands=2 olarak ayarlandı
detector = HandDetector(detectionCon=0.8, maxHands=2)

fingerTip = [4, 8, 12, 16, 20]
last_published_message = None # Son yayınlanan mesajı takip etmek için

# 10 parmak için renkler (El 1: 0-4, El 2: 5-9)
red = (0, 0, 255)
yellow = (0, 255, 255)
blue = (255, 0, 0)
green = (0, 255, 0)
purple = (255, 0, 255)
orange = (0, 165, 255)
pink = (203, 192, 255)
cyan = (255, 255, 0)
white = (255, 255, 255)
lime = (0, 255, 127)

colors = [red, yellow, blue, green, purple, # El 1 renkleri
          orange, pink, cyan, white, lime] # El 2 renkleri

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Kamera okunamadı.")
        break
        
    # Elleri bul (en fazla 2 el)
    hands, img = detector.findHands(img)
    
    currentHandsData = [""] * 2 # Bu frame'deki el verilerini tutmak için
    currentHandTypes = [""] * 2 # Bu frame'deki el tiplerini tutmak için
    num_hands = len(hands)

    # Algılanan her el için döngü
    for hand_idx, hand in enumerate(hands):
        lmList = hand['lmList']
        handType = hand['type']
        currentHandTypes[hand_idx] = handType # El tipini sakla
        
        # Mevcut el için parmak değerlerini hesapla
        current_fingerVal = [0] * 5

        # Başparmak (Thumb)
        if handType == "Right":
            if lmList[fingerTip[0]][0] > lmList[fingerTip[0] - 1][0]:
                current_fingerVal[0] = 1
            else:
                current_fingerVal[0] = 0
        else: # Left hand
            if lmList[fingerTip[0]][0] < lmList[fingerTip[0] - 1][0]:
                current_fingerVal[0] = 1
            else:
                current_fingerVal[0] = 0

        # Diğer 4 parmak
        for i in range(1, 5):
            if lmList[fingerTip[i]][1] < lmList[fingerTip[i] - 2][1]:
                current_fingerVal[i] = 1
            else:
                current_fingerVal[i] = 0

        # İşaretleri çiz (her parmak için ayrı renk)
        for i in range(5):
            if current_fingerVal[i] == 1:
                # El indeksine göre renk seçimi (hand_idx * 5 + i)
                color_index = hand_idx * 5 + i
                cv2.circle(img, (lmList[fingerTip[i]][0], lmList[fingerTip[i]][1]), 15,
                           colors[color_index], cv2.FILLED)

        # Mevcut elin parmak durumunu string'e çevir
        strVal = ''.join(map(str, current_fingerVal))
        currentHandsData[hand_idx] = strVal # Mevcut frame verisine ekle

    # --- El işleme döngüsü bitti, şimdi haritalama ve yayınlama ---
    
    message_to_publish = None

    if num_hands == 1:
        hand_type = currentHandTypes[0]
        str_val = currentHandsData[0]
        message_to_publish = hand_map.get(hand_type, {}).get(str_val)
    elif num_hands == 2:
        hand0_type = currentHandTypes[0]
        hand0_str = currentHandsData[0]
        hand1_type = currentHandTypes[1]
        hand1_str = currentHandsData[1]

        # Sol ve sağ eli belirle
        left_str = None
        right_str = None
        if hand0_type == 'Left' and hand1_type == 'Right':
            left_str = hand0_str
            right_str = hand1_str
        elif hand0_type == 'Right' and hand1_type == 'Left':
            left_str = hand1_str
            right_str = hand0_str

        if left_str is not None and right_str is not None:
            if left_str == right_str:
                lookup_key = f"{left_str}_{left_str}"
                message_to_publish = hand_map.get("Same", {}).get(lookup_key)
            else:
                lookup_key = f"{left_str}_{right_str}"
                message_to_publish = hand_map.get("Combined", {}).get(lookup_key)

    # Yayınlama Mantığı
    if message_to_publish is not None:
        if message_to_publish != last_published_message:
            print(f"Durum: {message_to_publish}")
            last_published_message = message_to_publish
    elif num_hands == 0: # Eller kaybolduysa
        if last_published_message is not None:
             print("Durum: El algılanmadı")
             last_published_message = None # Son mesajı sıfırla

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()