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
    ret = np.zeros((cmat.shape[0], cmat.shape[1])).astype(np.uint8)
    for i in range(cmat.shape[0]):
        for j in range(cmat.shape[1]):
            ret[i][j] = (0.21*cmat[i][j][0] + 0.72*cmat[i][j][1] + 0.07*cmat[i][j][2]).astype(np.uint8)

    return ret

# pads image by adding 1px, zero-valued border on all four sides,
# making an (n,n) image into an (n+2, n+2) image
def pad_cmatrix(cmat):
    return np.pad(cmat, ((1,1), (1,1), (0,0)), 'constant')

# same-pad image, i.e. for smaller of l,w, repeatedly add last row/column until l=w
def same_pad_cmatrix(cmat):
    if cmat.shape[0] < cmat.shape[1]:
        row = cmat[0]
        for i in range(cmat.shape[0], cmat.shape[1]):
            cmat = np.r_[cmat, [row]]
    elif cmat.shape[0] > cmat.shape[1]:
        col = cmat[:, -1]
        for i in range(cmat.shape[1], cmat.shape[0]):
            cmat = np.c_[cmat, col]
    return cmat





### MAIN CODE ###

img_filename = 'test1.png'

### IMAGE PRE-PROCESSING ACTIONS ###

# convert image to nxm matrix of (r,g,b) values
cmat = img_to_cmatrix(img_filename)

# convert color matrix to grayscale using: gray_val = 0.21*r + 0.72*g + 0.07*b
cmat = rgb_to_grayscale_cmatrix(cmat)

# same-pad image
cmat = same_pad_cmatrix(cmat)

# export pre-processed image
cmatrix_to_img(cmat, 'preprocessed_' + img_filename)
