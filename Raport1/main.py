from PIL import Image
import numpy as np

### utils
def print_np_arr(arr):
    Image.fromarray(arr).show()

img = Image.open('El-Capitan.jpg')
img = img.convert('RGB')

im = np.array(img)
negative = 255 - im

# Image.fromarray(negative).show()



### brightness
brightness = im + 10
# print_np_arr(bound(brightness))


### contrast 
contrast = bound(0.2*im + 10)
contrast = np.int_(contrast)
Image.fromarray(contrast.astype('uint8')).show()
