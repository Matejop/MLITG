import subprocess
import sys
import os

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
    pip_path = os.path.join("env", "Scripts", "pip.exe") if os.name == "nt" else os.path.join("env", "bin", "pip")
    print("📦 Installing requirements...")
    run(f"{pip_path} install -r requirements.txt")

    # 3. Run data deserialization + preprocess
    print("🧹 Preprocessing MNIST data...")
    python_path = os.path.join("env", "Scripts", "python.exe") if os.name == "nt" else os.path.join("env", "bin", "python")
    run(f"{python_path} data_preprocess.py")

    print("✅ All done! You can now start with your neural network 🚀")

if __name__ == "__main__":
    main()
