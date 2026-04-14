"""
Quantum Sniffer — Native Desktop GUI Launcher
=============================================
This script launches the Quantum Sniffer backend and wraps the Web Dashboard
into a native desktop application window.

Requirements:
  pip install PyQt6 PyQt6-WebEngine
"""

import sys
import threading
import time
import logging

try:
    from PyQt6.QtCore import QUrl
    from PyQt6.QtWidgets import QApplication, QMainWindow
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except ImportError:
    print("Error: PyQt6 or PyQt6-WebEngine is not installed.")
    print("Please run: pip install PyQt6 PyQt6-WebEngine")
    sys.exit(1)

def start_backend():
    try:
        from engine import CaptureEngine
        from web_dashboard import DashboardDataStore, start_web_dashboard
        
        # Initialize the global data store mapping
        web_store = DashboardDataStore()
        
        # Start the Flask web dashboard on port 5000
        start_web_dashboard(web_store, port=5000)
        
        # Initialize and start the packet capture engine
        # We disable the Rich terminal dashboard since we are using the GUI
        engine = CaptureEngine(
            interface=None,
            bpf_filter="",
            use_pqc=True,
            use_dashboard=False,
            sensitivity="medium",
            geoip=False
        )
        engine.start()
    except Exception as e:
        logging.error(f"Backend error: {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantum Sniffer Dashboard")
        self.setGeometry(100, 100, 1280, 850)
        
        # Set dark background color to match the dashboard
        self.setStyleSheet("QMainWindow { background-color: #06080f; }")
        
        # Create Web Engine View
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl("http://localhost:5000"))
        self.setCentralWidget(self.browser)

if __name__ == '__main__':
    print("Initializing Quantum Sniffer GUI...")
    
    # Start the backend engine in a background daemon thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Give the Flask server a brief moment to bind to the port
    time.sleep(1.5)
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    
    # Run the application event loop
    sys.exit(app.exec())
