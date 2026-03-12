import subprocess
import sys
import os

#TODO add filepaths to global config

PIP_PATH = os.path.join("env", "Scripts", "pip.exe") if os.name == "nt" else os.path.join("env", "bin", "pip")
PYTHON_PATH = os.path.join("env", "Scripts", "python.exe") if os.name == "nt" else os.path.join("env", "bin", "python")
PREPROCESS_SCRIPT_PATH = os.path.join("data", "data_preprocess.py")

def run(command):
    print(f"🟡 Running: {command}")
    subprocess.run(command, shell=True, check=True)

def main():
    # 1. Create virtual python environment
    if not os.path.isdir("env"):
        print("🧪 Creating virtual environment...")
        run(f"{sys.executable} -m venv env")
    else:
        print("✅ Virtual environment already exists.")

    # 2. Activate venv + install requirements
    print("📦 Installing requirements...")
    run(f"{PIP_PATH} install -r requirements.txt") #subprocess.run does not work without the use of os.path

    print("✅ All done! You can now start with your neural network 🚀")

if __name__ == "__main__":
    main()
