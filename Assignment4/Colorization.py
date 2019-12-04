from PIL import Image
import numpy as np


### IMAGE PRE-PROCESSING UTILITY FUNCTIONS ###

# converts image to matrix of color values
def img_to_cmatrix(filename):
    img = Image.open('./imgs/' + filename)
    cmat = np.array(img)
    return cmat

# converts matrix to image and saves to ./imgs/
def cmatrix_to_img(cmat, filename):
    img = Image.fromarray(cmat)
    img.save('./imgs/' + filename)
    # img.show()

# crops color matrix to square based on smaller of l, w
def crop_cmatrix(cmat):
    return cmat[0:min(cmat.shape[0], cmat.shape[1]), 0:min(cmat.shape[0], cmat.shape[1])]

# converts RGB color matrix to grayscale matrix
def rgb_to_grayscale_cmatrix(cmat):
    for i in range(cmat.shape[0]):
        for j in range(cmat.shape[1]):
            cmat[i][j] = 0.21*cmat[i][j][0] + 0.72*cmat[i][j][1] + 0.07*cmat[i][j][2]

    return cmat






### MAIN CODE ###

img_filename = 'test1.png'


### IMAGE PRE-PROCESSING ACTIONS ###

# convert image to nxm matrix of (r,g,b) values
cmat = img_to_cmatrix(img_filename)

# crop nxm color matrix to nxn where n<=m
cmat = crop_cmatrix(cmat)

# convert color matrix to grayscale using: gray_val = 0.21*r + 0.72*g + 0.07*b
cmat = rgb_to_grayscale_cmatrix(cmat)

# export pre-processed image
cmatrix_to_img(cmat, 'preprocessed_' + img_filename)
