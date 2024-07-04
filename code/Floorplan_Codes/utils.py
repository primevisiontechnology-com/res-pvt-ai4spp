import os
from PIL import Image

def get_paths(name, path, recursive=True, depth=1):
    results = []
    for root, dirs, files in os.walk(path):
        if name in files:
            results.append(os.path.join(root, name))
        if not recursive:
            return results
        if depth > 0:
            for dir in dirs:
                result = get_paths(name, os.path.join(root, dir), recursive, depth - 1)
    return results

def gen_pdf(paths, prefix="./", suffix=".png"):
    images = []
    for path in paths:
        images.append(Image.open(os.path.join(prefix, path+suffix)))
    pdf_path = prefix+"fp-catalog.pdf"
    images[0].save(pdf_path, save_all=True, append_images=images[1:])

