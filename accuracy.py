# prints confusion matrix, overall accuracy, kappa to a file

# import required modules
from PIL import Image
import numpy as np

image = "Data/CLASSIFIED IMAGE.TIF"

# define color constants
class1color = (0, 119,204)  # water
class2color = (0,170,170)  # shallow water
class3color = (76, 153, 0)  # mangrove forest
class4color = (210, 210,50)  #open land
class5color = (255, 255, 255)  # clouds
class6color = (153,76,0)  #dispersed settlements
class7color = (221, 170, 0)  #nucleated settlements

color_list = {class1color: 0, class2color: 1, class3color: 2, class4color: 3,
              class5color: 4, class6color: 5, class7color: 6}

# read training data
index = np.genfromtxt("Data/TRAINING WINDOW INDEXES.dat")
index = index.astype(int)

# load the image in RGB mode
im = Image.open(image).convert('RGB')

# calculate confusion matrix
con = np.zeros((7, 7))
for a in range(7):
    for b in range(index[a][0], index[a][2]):
        for c in range(index[a][1], index[a][3]):
            (r1, g1, b1) = im.getpixel((b, c))
            con[color_list[(r1, g1, b1)], a] += 1

# calculate Overall accuracy and Kappa Coefficient
horizontal = con.sum(axis=0)
vertical = con.sum(axis=1)
pix_count = horizontal.sum(axis=0)

oa = 0  # calculate observed agreement/overall accuracy
for i in range(7):
    oa += con[i, i] / pix_count

ra = 0  # calculate random agreement
for i in range(7):
    ra += (horizontal[i] * vertical[i]) / (pix_count * pix_count)
kappa = (oa - ra) / (1 - ra)  # cohen's kappa

# print the result to console
for i in range(7):
    print(con[i])
    
print("\nCohen's kappa : ", kappa)
print("\nOverall accuracy : ", oa)
