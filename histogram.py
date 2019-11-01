# import required modules
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# read training data
index = np.genfromtxt("Data/TRAINING WINDOW INDEXES.dat")
index = index.astype(int)
color=['blue','maroon','orange','brown','skyblue','green','red']
image = Image.open("Data/BLUE BAND.TIF").convert('L')
plt.xlabel('PIXEL INTENSITY')
plt.ylabel('PIXEL COUNT')
plt.axis([0, 256, 0, 100])
for m in range(0, 7):
    crop_rectangle = (index[m][0], index[m][1], index[m][2], index[m][3])
    cropped_im = image.crop(crop_rectangle)
    hist = cropped_im.histogram()
    plt.title('Pixel intensity distribution of class '+str(m+1))
    plt.bar(range(0,256), hist,color=color[m])
    plt.show()
