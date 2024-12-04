import os
from torchvision import transforms
import shutil


def make_folder(path, folder_name, replace=False):
    dir = os.path.join(path, folder_name)
    if replace is True:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        
    if not os.path.exists(dir):
            os.makedirs(dir)

def tensor_to_PIL(img):
    img = transforms.functional.to_pil_image(img)
    return img

def denorm(x):
    out = (x + 1.0) / 2.0
    return out.clamp_(0, 1)


