#!/usr/bin/env python3
"""
SentryBOT Desktop GUI
Kullanım:
    python desk_gui.py --robot-ip 192.168.137.52 --debug
"""

import sys
from PyQt5.QtWidgets import QApplication
from modules.gui.desk_gui_app import DeskGUI
import multiprocessing
import traceback

def launch_gui(**kwargs):
    """
    SentryBOT GUI uygulamasını başlat
    
    Args:
        **kwargs: DeskGUI sınıfına geçirilecek parametreler
            robot_ip: Robot IP adresi
            video_port: Video portu
            command_port: Komut portu
            ollama_url: Ollama API URL
            ollama_model: Kullanılacak model
            encodings_file: Yüz tanıma modeli dosyası
            bluetooth_server: Bluetooth ses sunucusu
            debug: Debug modu
    """
    try:
        # On Windows, ensure that the multiprocessing functionality works
        if sys.platform.startswith('win'):
            multiprocessing.freeze_support()
        
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        # Create and show the GUI
        window = DeskGUI(**kwargs)
        window.show()
        
        # Start the application event loop
        sys.exit(app.exec_())
    except Exception as e:
        print(f"GUI başlatılırken hata oluştu: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SentryBOT Masaüstü GUI')
    parser.add_argument('--robot-ip', default='192.168.137.52', help='Robot IP adresi')
    parser.add_argument('--video-port', type=int, default=8000, help='Video portu')
    parser.add_argument('--command-port', type=int, default=8090, help='Komut portu')
    parser.add_argument('--gui-listen-port', type=int, default=8091, help='GUI dinleme portu')
    parser.add_argument('--ollama-url', default='http://localhost:11434/api', help='Ollama API URL')
    parser.add_argument('--ollama-model', default='gemma2:2b', help='Kullanılacak model')
    parser.add_argument('--encodings-file', default='encodings.pickle', help='Yüz tanıma modeli dosyası')
    parser.add_argument('--bluetooth-server', default='192.168.1.100', help='Bluetooth ses sunucusu')
    parser.add_argument('--debug', action='store_true', help='Debug modunu aktif et')
    
    args = parser.parse_args()
    
    config = vars(args)
    launch_gui(**config)