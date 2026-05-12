from pdf2image import convert_from_path
import os


def convert_to_images(pdf_path: str, output_dir: str, dpi: int = 300) -> list[str]:
    """Convert PDF pages to high-resolution images."""
    images = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []
    for i, img in enumerate(images):
        path = os.path.join(output_dir, f"page_{i + 1}.png")
        img.save(path, "PNG")
        image_paths.append(path)
    return image_paths
