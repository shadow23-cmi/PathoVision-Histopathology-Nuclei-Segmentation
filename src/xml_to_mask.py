from pathlib import Path
import xml.etree.ElementTree as ET
from utils import PROJECT_ROOT, RAW_DATA_DIR, IMG_DATA_DIR, XML_DATA_DIR , IMG_DIR, MASK_DIR

import cv2
import numpy as np
from PIL import Image



def find_image_for_xml(xml_path):
    """
    Find the corresponding image.

    First try the same filename with .tif.
    If that does not exist, try matching by stem.
    """
    candidate = xml_path.with_suffix(".tif")

    if candidate.exists():
        return candidate

    matches = list(RAW_DATA_DIR.rglob(f"{xml_path.stem}.tif"))

    if len(matches) == 1:
        return matches[0]

    return None


def parse_xml_to_mask(xml_path, image_shape):
    """
    Convert MoNuSeg XML polygon annotations
    into a binary mask.

    image_shape: (height, width)
    """
    height, width = image_shape

    mask = np.zeros((height, width), dtype=np.uint8)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # MoNuSeg XML structure:
    # Annotations -> Annotation -> Coordinates -> Coordinate
    for annotation in root.iter("Annotation"):

        for region in annotation.iter("Region"):
            coordinates = []

            for coordinate in region.iter("Vertex"):

                x = float(coordinate.attrib["X"])
                y = float(coordinate.attrib["Y"])

                # Round coordinates to nearest pixel.
                x = int(round(x))
                y = int(round(y))

                # Clip to image bounds.
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))

                coordinates.append([x, y])

            if len(coordinates) >= 3:

                polygon = np.array(
                    coordinates,
                    dtype=np.int32
                )

                cv2.fillPoly(
                    mask,
                    [polygon],
                    color=1
                )

    return mask


def main():

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    xml_files = list(RAW_DATA_DIR.rglob("*.xml"))

    print(f"Found {len(xml_files)} XML files")

    converted = 0
    skipped = 0

    for xml_path in xml_files:

        image_path = find_image_for_xml(xml_path)

        if image_path is None:
            print(f"Skipping: no matching image for {xml_path.name}")
            skipped += 1
            continue

        # Read image only to obtain height and width.
        image = Image.open(image_path)

        width, height = image.size

        mask = parse_xml_to_mask(
            xml_path,
            image_shape=(height, width)
        )

        output_path = MASK_DIR / f"{image_path.stem}.png"

        # Save image.
        output_image_path = IMG_DIR / f"{image_path.stem}.png"

        if not output_image_path.exists():
            image.save(output_image_path)

        # Save binary mask as 0 and 255.
        mask_uint8 = (mask * 255).astype(np.uint8)

        Image.fromarray(mask_uint8).save(output_path)

        converted += 1

        print(
            f"[{converted}] "
            f"{image_path.name} -> {output_path.name}"
        )

    print()
    print(f"Converted: {converted}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()