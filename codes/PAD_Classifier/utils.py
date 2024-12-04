import os
import shutil

def make_folder(path, folder_name, replace=False):
    dir = os.path.join(path, folder_name)
    if replace is True:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        
    if not os.path.exists(dir):
            os.makedirs(dir)
    