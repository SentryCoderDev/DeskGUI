import cv2
import os
import json
import time
import random
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from pubsub import pub
from ..vision import get_model_path

class FingerTracking:
    def __init__(self, command_sender=None):
        """El takibi modülü, parmak hareketlerini algılar ve komutları yayınlar."""
        self.command_sender = command_sender
        self.processing_active = False
        
        # El takipçisi ayarları
        self.detector = HandDetector(detectionCon=0.8, maxHands=2)
        self.fingerTip = [4, 8, 12, 16, 20]  # Parmak uçlarının indeksleri
        
        # Komut haritasını yükle
        self.hand_map = self._load_hand_map()
        
        # Motto kümesi: rastgele tek kelimeler
        self.motto_list = [
            "Başarı", "Azim", "Odak", "Sabır", "Güç", "İnanç", "Cesaret", "Bilgi", "Deneyim", "Yaratıcılık"
        ]
        self.argo_list = ["lan", "moruk", "kaptan", "reis", "dayı"]
        
        # Son komut durumu
        self.last_published_command = None
        self.min_command_interval = 0.5  # Komutlar arası minimum süre (saniye)
        self.last_command_time = 0
        
        # İşleme durumu
        self.processing_active = False
        self.log("FingerTracking modülü başlatıldı")

    def _load_hand_map(self):
        """El hareketleri-komut haritasını yükle."""
        return {
            "Right": {
                "00000": "stop_tts",
                "10000": "servo_zero",
                "01000": "wave_animation",
                "00100": "argo_sentence_deniz",
                "00010": "say_motto_evolution",
                "00001": "breathe_anim",
                "01111": "play_music",
                "11111": "head_right",
                "11000": "finger_combo_one",
                "10100": "finger_combo_two"
            },
            "Left": {
                "00000": "stop_tts",
                "10000": "servo_ninety",
                "01000": "stacked_bars_animation",
                "00100": "argo_sentence_mali",
                "00010": "say_motto_death",
                "00001": "breathe_anim",
                "01111": "play_music",
                "11111": "head_left",
                "11000": "finger_combo_three",
                "10100": "finger_combo_four"
            },
            "Same": {
                "00000_00000": "double_reset",
                "01111_01111": "play_music_bass",
                "11000_11000": "double_finger_combo"
            },
            "Combined": {
                "01000_01000": "introduce",
                "00001_00001": "bye",
                "11000_00111": "special_combo"
            }
        }

    def print_all_combinations(self):
        """Tüm el kombinasyonlarını ve karşılık gelen komutları yazdırır."""
        for hand_type, combos in self.hand_map.items():
            print(f"\n{hand_type} kombinasyonları:")
            for combo, command in combos.items():
                print(f"  {combo} -> {command}")

    def start(self):
        """İşlemeyi başlat."""
        self.processing_active = True
        self.log("FingerTracking başlatıldı")
        return True

    def stop(self):
        """İşlemeyi durdur."""
        self.processing_active = False
        self.log("FingerTracking durduruldu")

    def process_frame(self, frame):
        """Bir kareyi işle ve el durumunu tespit et."""
        if not self.processing_active or frame is None:
            return frame, None

        try:
            # Kareyi kopyala
            processed_frame = frame.copy()
            
            # Elleri bul
            hands, img = self.detector.findHands(processed_frame)
            
            currentHandsData = [""] * 2  # Bu frame'deki el verilerini tutmak için
            currentHandTypes = [""] * 2  # Bu frame'deki el tiplerini tutmak için
            num_hands = len(hands)

            # Algılanan her el için döngü
            for hand_idx, hand in enumerate(hands):
                lmList = hand['lmList']
                handType = hand['type']
                currentHandTypes[hand_idx] = handType  # El tipini sakla
                
                # Mevcut el için parmak değerlerini hesapla
                current_fingerVal = [0] * 5

                # Başparmak (Thumb)
                if handType == "Right":
                    if lmList[self.fingerTip[0]][0] > lmList[self.fingerTip[0] - 1][0]:
                        current_fingerVal[0] = 1
                    else:
                        current_fingerVal[0] = 0

                # Diğer 4 parmak
                for i in range(1, 5):
                    if lmList[self.fingerTip[i]][1] < lmList[self.fingerTip[i] - 2][1]:
                        current_fingerVal[i] = 1
                    else:
                        current_fingerVal[i] = 0

                # Çizimleri yap (her parmak için)
                for i in range(5):
                    if current_fingerVal[i] == 1:
                        # Parmak kalkıksa işaretle
                        cv2.circle(img, (lmList[self.fingerTip[i]][0], lmList[self.fingerTip[i]][1]), 
                                  15, (0, 255, 0), cv2.FILLED)

                # Mevcut elin parmak durumunu string'e çevir
                strVal = ''.join(map(str, current_fingerVal))
                currentHandsData[hand_idx] = strVal  # Mevcut frame verisine ekle

            # El verilerine göre komutu belirle
            command = self._determine_command(num_hands, currentHandTypes, currentHandsData)
            
            # Komut bulunduysa ve son komuttan belirli süre geçtiyse yayınla
            current_time = time.time()
            if command and (current_time - self.last_command_time > self.min_command_interval):
                if command != self.last_published_command:
                    self.publish_command(command)
                    self.last_published_command = command
                    self.last_command_time = current_time
            
            # Eller kaybolduysa ve önceden komut yayınlanmışsa
            if num_hands == 0 and self.last_published_command is not None:
                # Minimum komut aralığı geçtiyse sıfırla
                if current_time - self.last_command_time > self.min_command_interval:
                    self.last_published_command = None
                    self.publish_command("reset")
                    self.last_command_time = current_time

            return img, {"command": self.last_published_command}
            
        except Exception as e:
            self.log(f"Kare işlenirken hata: {e}")
            return frame, None

    def _determine_command(self, num_hands, hand_types, hands_data):
        """El verilerine göre hangi komutun yayınlanacağını belirler."""
        if num_hands == 0:
            return None

        if num_hands == 1:
            hand_type = hand_types[0]
            str_val = hands_data[0]
            # Sağ, sol veya başka bir el tipi için haritadan komut bul
            return self.hand_map.get(hand_type, {}).get(str_val)

        elif num_hands == 2:
            hand0_type = hand_types[0]
            hand0_str = hands_data[0]
            hand1_type = hand_types[1]
            hand1_str = hands_data[1]

            # Sol ve sağ eli belirle
            left_str = right_str = None
            if hand0_type == 'Left' and hand1_type == 'Right':
                left_str = hand0_str
                right_str = hand1_str
            elif hand0_type == 'Right' and hand1_type == 'Left':
                left_str = hand1_str
                right_str = hand0_str

            # Eğer iki el de aynı tip ve aynı kombinasyonda ise "Same" haritasını kullan
            if hand0_type == hand1_type and hand0_str == hand1_str:
                lookup_key = f"{hand0_str}_{hand1_str}"
                return self.hand_map.get("Same", {}).get(lookup_key)

            # Eğer biri sol biri sağ ise "Combined" haritasını kullan
            if left_str is not None and right_str is not None:
                lookup_key = f"{left_str}_{right_str}"
                return self.hand_map.get("Combined", {}).get(lookup_key)

        return None

    def publish_command(self, command):
        """Tespit edilen komutu pubsub ile yayınlar ve isteğe bağlı olarak robota gönderir."""
        if not command:
            return
            
        self.log(f"Parmak komutu algılandı: {command}")
        
        # PubSub ile komutu yayınla
        pub.sendMessage('gesture_command', command=command)
        
        # Komut gönderici varsa robota gönder
        if self.command_sender and self.command_sender.connected:
            try:
                params = {}
                command_to_send = command

                if command == 'play_music':
                    params = {'path': 'music/chill_vibes.mp3'}
                elif command == 'play_music_bass':
                    params = {'path': 'music/bass_boosted_beats.mp3'}
                    command_to_send = 'play_music'
                elif command.startswith('say_motto'):
                    motto = random.choice(self.motto_list)
                    params = {'text': motto}
                    command_to_send = 'tts'
                elif command.startswith('argo_sentence'):
                    target = command.split('_')[-1]
                    sentence = f"{target.capitalize()} {random.choice(self.argo_list)}"
                    params = {'text': sentence}
                    command_to_send = 'tts'
                elif command == 'servo_zero':
                    params = {'identifier': 'pan', 'percentage': 0, 'absolute': True}
                    command_to_send = 'servo_move'
                elif command == 'servo_ninety':
                    params = {'identifier': 'pan', 'percentage': 90, 'absolute': True}
                    command_to_send = 'servo_move'
                elif command == 'wave_animation':
                    params = {'animation': 'WAVE_HAND'}
                    command_to_send = 'send_animation'
                elif command == 'stacked_bars_animation':
                    params = {'animation': 'STACKED_BARS'}
                    command_to_send = 'send_animation'
                elif command == 'breathe_anim':
                    params = {'animation': 'BREATHE', 'color': 'BLUE'}
                    command_to_send = 'send_animation'
                elif command == 'head_right':
                    params = {'animation': 'HEAD_RIGHT'}
                    command_to_send = 'send_animation'
                elif command == 'head_left':
                    params = {'animation': 'HEAD_LEFT'}
                    command_to_send = 'send_animation'      
                    
                elif command.startswith('finger_combo'):
                    combo_type = command.split('_')[-1]
                    params = {'animation': f'FINGER_COMBO_{combo_type.upper()}', 'color': 'GREEN'}
                    command_to_send = 'send_animation'
                elif command == 'double_finger_combo':
                    params = {'animation': 'DOUBLE_FINGER_COMBO', 'color': 'PURPLE'}
                    command_to_send = 'send_animation'
                elif command == 'special_combo':
                    params = {'animation': 'SPECIAL_COMBO', 'color': 'GOLD'}

                self.command_sender.send_command(command_to_send, params)
            except Exception as e:
                self.log(f"Komut gönderilirken hata: {e}")

    def log(self, message):
        """Loglama işlemlerini yapar."""
        pub.sendMessage('log', msg=f"[FingerTracking] {message}")
