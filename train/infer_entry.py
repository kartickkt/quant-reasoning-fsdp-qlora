import subprocess
import sys

if __name__ == "__main__":
    subprocess.check_call([sys.executable, "compare_inference.py"])