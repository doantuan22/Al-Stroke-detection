"""
Stroke AI - Main Entry Point
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent))

from app.gui.main_window import StrokeApp

def main():
    print("="*60)
    print("AI Stroke")
    print("="*60)
    
    # Initialize application
    app = StrokeApp()
    
    # Start main loop
    app.mainloop()

if __name__ == "__main__":
    main()
