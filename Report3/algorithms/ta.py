from PIL import Image
import numpy as np
import os

def make_dir(dir_name):
    i = ''
    if os.path.exists(dir_name):
        i = 1
        while os.path.exists(f'{dir_name}{i}'):
            i += 1
    
    os.makedirs(f'{dir_name}{i}')
    return f'{dir_name}{i}'

class ThinningAlgorithm:
    def __init__(self, img_arr, show_steps, save_steps, save_location, dir_name):
        self.img_arr = img_arr
        self.show_steps = show_steps
        self.save_steps = save_steps
        folder_name = make_dir(save_location + dir_name)
        self.save_location = folder_name + '/'

    def debug(self, name):
        max_val = np.amax(self.img_arr)
        max_val = max_val if max_val >= 1 else 1
        factor = 255 / max_val
        img = Image.fromarray(self.img_arr.astype(np.int8)*factor)
        if self.show_steps:
            img.show()
        if self.save_steps:
            img = img.convert('L')
            img.save(self.save_location + name + '.png')
    
