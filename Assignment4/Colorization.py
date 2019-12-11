from PIL import Image
import numpy as np
from skimage import io, color

n = 64


### IMAGE PRE-PROCESSING UTILITY FUNCTIONS ###

# converts image to matrix of color values
def imgs_to_cmatrices(filenames):
    rets = []
    for filename in filenames:
        img = Image.open('./imgs/' + filename)
        tmp = np.array(img)

        ret = np.zeros((n, n, 3))

        if tmp.shape[0] >= n and tmp.shape[1] >= n:
            for i in range(n):
                for j in range(n):
                    ret[i][j] = np.array([tmp[i][j][0], tmp[i][j][1], tmp[i][j][2]])

            rets.append(ret)
        else:
            return None

    return np.stack(rets, axis=0)

# converts matrix to image and saves to ./imgs/
def cmatrices_to_imgs(cmats, filenames):
    i = 0
    for filename in filenames:
        cmat = cmats[i, 0, :, :].astype(np.uint8)
        img = Image.fromarray(cmat)
        img.save('./imgs/' + 'preprocessed_' + filename)
        i += 1

# crops color matrix to square based on smaller of l, w
def crop_cmatrices(cmats):
    rets = np.zeros((cmats.shape[0], cmats.shape[1], cmats.shape[2], cmats.shape[3]))
    for i in range(cmats.shape[0]):
        cmat = cmats[i, :, :, :]
        rets.append(cmat[0:min(cmat.shape[0], cmat.shape[1]), 0:min(cmat.shape[0], cmat.shape[1])])
    return rets

# converts RGB color matrix to grayscale matrix
def rgb_to_grayscale_cmatrices(cmats):
    rets = []
    for i in range(len(cmats)):
        cmat = cmats[i, :, :, :]
        ret = np.zeros((1, cmat.shape[0], cmat.shape[1])).astype(np.uint8)
        for i in range(cmat.shape[0]):
            for j in range(cmat.shape[1]):
                ret[0][i][j] = 0.21*cmat[i][j][0] + 0.72*cmat[i][j][1] + 0.07*cmat[i][j][2]
        rets.append(ret)
    return np.stack(rets, axis=0)

# converts RGB color matrix to LAB matrix
def rgb_to_lab_cmatrices(cmats):
    rets = []
    for i in range(len(cmats)):
        cmat = cmats[i, :, :, :]
        tmp = cmat / 255
        rets.append((color.rgb2lab(tmp)))
    return np.stack(rets, axis=0)

# converts LAB matrix to color matrix
def lab_to_rgb_cmatrices(cmats):
    rets = []
    for i in range(len(cmats)):
        cmat = cmats[i, :, :, :]
        rets.append(color.lab2rgb(cmat) * 255)
    return np.stack(rets, axis=0)

# converts LAB matrix to grayscale matrix
def lab_to_grayscale_cmatrices(cmats):
    rets = []
    for i in range(len(cmats)):
        cmat = cmats[i, :, :, :]
        ret = np.zeros((cmat.shape[0], cmat.shape[1]))
        for i in range(cmat.shape[0]):
            for j in range(cmat.shape[0]):
                ret[i][j] = cmat[i][j][0]
        rets.append(ret)
    return np.stack(rets, axis=0)

# converts grayscale matrix to LAB matrix
def grayscale_to_lab_cmatrices(cmats):
    rets = []
    for i in range(len(cmats)):
        cmat = cmats[i, :, :, :]
        ret = np.zeros((cmat.shape[0], cmat.shape[1], 3))
        for i in range(cmat.shape[0]):
            for j in range(cmat.shape[0]):
                ret[i][j] = [cmat[i][j], 0, 0]
        rets.append(ret)
    return np.stack(rets, axis=0)

# pads image by adding 1px, zero-valued border on all four sides,
# making an (n,n) image into an (n+2, n+2) image
def pad_cmatrices(cmats):
    rets = []
    for i in range(len(cmats)):
        cmat = cmats[i, :, :, :]
        rets.append(np.pad(cmat, ((1,1), (1,1), (0,0)), 'constant'))
    return np.stack(rets, axis=0)

# same-pad image, i.e. for smaller of l,w, repeatedly add last row/column until l=w
def same_pad_cmatrices(cmats):
    rets = []
    for i in range(len(cmats)):
        cmat = cmats[i, :, :, :]
        if cmat.shape[0] < cmat.shape[1]:
            row = cmat[0]
            for i in range(cmat.shape[0], cmat.shape[1]):
                cmat = np.r_[cmat, [row]]
            rets.append(cmat)
        elif cmat.shape[0] > cmat.shape[1]:
            col = cmat[:, -1]
            for i in range(cmat.shape[1], cmat.shape[0]):
                cmat = np.c_[cmat, col]
            rets.append(cmat)
    return np.stack(rets, axis=0)



'''
### MAIN CODE ###

filenames = ['test1.png', 'test2.jpeg', 'test3.png', 'test4.jpg']

### IMAGE PRE-PROCESSING ACTIONS ###

# convert image to nxm matrix of (r,g,b) values
cmats = imgs_to_cmatrices(filenames)

# convert color matrix to grayscale using: gray_val = 0.21*r + 0.72*g + 0.07*b
cmats = rgb_to_grayscale_cmatrices(cmats)

# same-pad image
#cmats = same_pad_cmatrices(cmats)

# export pre-processed image
cmatrices_to_imgs(cmats, filenames)

#cmats = rgb_to_lab_cmatrices(cmats)

#cmats = lab_to_grayscale_cmatrices(cmats)

'''