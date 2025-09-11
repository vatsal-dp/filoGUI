#!/usr/bin/env python3
"""
Development runner with hot reload for the GUI application.
Usage: python dev_runner.py
"""

import os
import sys
import subprocess
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class CodeChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self.process = None
        self.restart_app()
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        if event.src_path.endswith('.py'):
            print(f"🔄 Detected change in: {event.src_path}")
            self.restart_app()
    
    def restart_app(self):
        # Kill existing process
        if self.process and self.process.poll() is None:
            print("🛑 Stopping existing application...")
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                
        # Start new process
        print("🚀 Starting application...")
        self.process = subprocess.Popen([sys.executable, "main.py"])
        print("✅ Application started!")

def main():
    print("🔥 Hot Reload Development Mode")
    print("📁 Watching current directory for Python file changes...")
    print("Press Ctrl+C to stop")
    
    event_handler = CodeChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, ".", recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping hot reload...")
        observer.stop()
        if event_handler.process and event_handler.process.poll() is None:
            event_handler.process.terminate()
            
    observer.join()
    print("👋 Hot reload stopped")

if __name__ == "__main__":
    main()