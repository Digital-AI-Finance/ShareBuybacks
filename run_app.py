"""
Launcher script for PyInstaller packaging of Streamlit app.

This script starts the Streamlit server programmatically, allowing
the app to be packaged as a standalone executable.

Supports both Windows and macOS platforms.
"""

import sys
import os
import subprocess
import webbrowser
import threading
import time
import platform
from streamlit.web import cli as stcli


def kill_port_8501():
    """Kill any process using port 8501. Cross-platform support."""
    try:
        if platform.system() == 'Windows':
            # Windows: use netstat and taskkill
            result = subprocess.run(
                'netstat -ano | findstr :8501',
                shell=True, capture_output=True, text=True
            )
            killed_pids = set()
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        if pid.isdigit() and pid not in killed_pids:
                            subprocess.run(
                                f'taskkill /F /PID {pid}',
                                shell=True, capture_output=True
                            )
                            killed_pids.add(pid)
        else:
            # macOS/Linux: use lsof and kill
            result = subprocess.run(
                'lsof -ti:8501',
                shell=True, capture_output=True, text=True
            )
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid.strip().isdigit():
                    subprocess.run(
                        f'kill -9 {pid}',
                        shell=True, capture_output=True
                    )
    except Exception:
        pass  # Ignore errors silently


def open_browser_delayed():
    """Open browser after a short delay to allow server startup."""
    time.sleep(3)
    webbrowser.open('http://localhost:8501')


def main():
    """Launch the Streamlit application."""
    # Kill any existing process on port 8501
    kill_port_8501()

    # Get the directory where the executable is located
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_path = sys._MEIPASS
        app_path = os.path.join(base_path, 'app.py')
    else:
        # Running as script
        base_path = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(base_path, 'app.py')

    # Configure Streamlit arguments
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--server.port=8501",
        "--theme.base=light"
    ]

    # Start browser in background thread
    browser_thread = threading.Thread(target=open_browser_delayed, daemon=True)
    browser_thread.start()

    # Start Streamlit (blocks until server stops)
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
