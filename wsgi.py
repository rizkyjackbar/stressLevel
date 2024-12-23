import sys
import os

# Menentukan path ke folder aplikasi
sys.path.insert(0, os.path.dirname(__file__))

# Mengimpor aplikasi Flask dari app.py
from app import app as application