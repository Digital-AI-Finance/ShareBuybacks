"""
Build script for creating a minimal PyInstaller executable.

This script:
1. Creates a fresh minimal virtual environment
2. Installs only required dependencies
3. Builds the executable using PyInstaller
"""

import subprocess
import sys
import os
import shutil

# Configuration
VENV_DIR = "venv_minimal"
SPEC_FILE = "ShareBuybackApp.spec"

# Minimal dependencies - only what the app actually needs
DEPENDENCIES = [
    "streamlit>=1.29.0",
    "plotly>=5.18.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "altair>=5.0.0",
    "pyarrow>=14.0.0",  # Required by streamlit for dataframes
    "pyinstaller>=6.0.0",
]


def run_cmd(cmd, desc=""):
    """Run a command and show output."""
    print(f"\n{'='*60}")
    print(f">>> {desc or cmd}")
    print('='*60)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: Command failed with code {result.returncode}")
        sys.exit(1)
    return result


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    print("\n" + "="*60)
    print("MINIMAL BUILD: Share Buyback Strategy App")
    print("="*60)

    # Step 1: Clean up old build artifacts
    print("\n[1/5] Cleaning up old build artifacts...")
    for folder in ["build", "dist", VENV_DIR]:
        if os.path.exists(folder):
            print(f"  Removing {folder}/")
            shutil.rmtree(folder)

    # Step 2: Create minimal virtual environment
    print(f"\n[2/5] Creating minimal virtual environment: {VENV_DIR}")
    run_cmd(f"python -m venv {VENV_DIR}", "Creating virtual environment")

    # Determine python path based on OS
    if sys.platform == "win32":
        python_path = os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        python_path = os.path.join(VENV_DIR, "bin", "python")

    # Step 3: Upgrade pip (use -m pip to avoid locking issues on Windows)
    print("\n[3/5] Upgrading pip...")
    run_cmd(f'"{python_path}" -m pip install --upgrade pip', "Upgrading pip")

    # Step 4: Install minimal dependencies
    print("\n[4/5] Installing minimal dependencies...")
    deps_str = " ".join([f'"{dep}"' for dep in DEPENDENCIES])
    run_cmd(f'"{python_path}" -m pip install {deps_str}', "Installing dependencies")

    # Step 5: Build with PyInstaller
    print("\n[5/5] Building executable with PyInstaller...")
    run_cmd(f'"{python_path}" -m PyInstaller {SPEC_FILE} --clean', "Running PyInstaller")

    # Check result
    exe_path = os.path.join("dist", "ShareBuybackStrategy.exe")
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print("\n" + "="*60)
        print("BUILD SUCCESSFUL!")
        print("="*60)
        print(f"Executable: {os.path.abspath(exe_path)}")
        print(f"Size: {size_mb:.1f} MB")
        print("\nTo run: Double-click the .exe or run from command line")
        print("The app will open in your default browser at http://localhost:8501")
    else:
        print("\nERROR: Executable not found!")
        sys.exit(1)


if __name__ == "__main__":
    main()
