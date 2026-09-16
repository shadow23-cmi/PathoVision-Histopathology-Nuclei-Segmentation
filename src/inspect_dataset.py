from utils import PROJECT_ROOT, RAW_DATA_DIR, IMG_DATA_DIR, XML_DATA_DIR


def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {RAW_DATA_DIR}")
    print()

    if not RAW_DATA_DIR.exists():
        print("Data directory does not exist.")
        print("Download and extract MoNuSeg first.")
        return

    image_files = list(RAW_DATA_DIR.rglob("*.tif"))
    xml_files = list(RAW_DATA_DIR.rglob("*.xml"))

    print(f"PNG files: {len(image_files)}")
    print(f"XML files: {len(xml_files)}")
    print()

    print("First 10 PNG files:")
    for path in image_files[:10]:
        print(path.relative_to(PROJECT_ROOT))

    print()

    print("First 10 XML files:")
    for path in xml_files[:10]:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()