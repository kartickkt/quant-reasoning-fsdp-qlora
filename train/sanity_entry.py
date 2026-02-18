import subprocess
import sys

if __name__ == "__main__":
    subprocess.check_call([sys.executable, "lora_sanity_check.py"])