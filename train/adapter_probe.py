import os
import tarfile

ADAPTER_PATH = "/opt/ml/input/data/adapter"


def extract_adapter():
    tar_path = os.path.join(ADAPTER_PATH, "model.tar.gz")

    print("\n=== Checking for tar.gz ===")
    print("Tar exists:", os.path.exists(tar_path))

    if os.path.exists(tar_path):
        print("\n=== Extracting tar.gz ===")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(ADAPTER_PATH)
        print("Extraction complete.")


def walk_dir(root):
    print(f"\n=== Directory tree: {root} ===")
    for dirpath, dirnames, filenames in os.walk(root):
        level = dirpath.replace(root, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(dirpath)}/")

        subindent = "  " * (level + 1)
        for f in filenames:
            print(f"{subindent}{f}")


def main():
    print("\n=== ADAPTER ROOT ===")
    print("Path:", ADAPTER_PATH)

    print("\n=== Before extraction ===")
    walk_dir(ADAPTER_PATH)

    extract_adapter()

    print("\n=== After extraction ===")
    walk_dir(ADAPTER_PATH)


if __name__ == "__main__":
    main()