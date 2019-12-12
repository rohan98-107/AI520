from PIL import Image
import numpy as np

### IMAGE PRE-PROCESSING UTILITY FUNCTIONS ###

# converts image to matrix of color values
def imgs_to_cmatrices(filenames, n):
    rets = np.zeros((len(filenames), 3, n, n))
    for f in range(len(filenames)):
        img = Image.open('./imgs/' + filenames[f])
        img = img.resize((n, n), Image.ANTIALIAS)
        tmp = np.array(img)
        for i in range(n):
            for j in range(n):
                rets[f, 0, i, j] = tmp[i][j][0]
                rets[f, 1, i, j] = tmp[i][j][1]
                rets[f, 2, i, j] = tmp[i][j][2]
    return rets

# converts matrix to image and saves to ./imgs/
# can handle cmats (m x 3 x n x n) or gmats (m x 1 x n x n)
def cmatrices_to_imgs(cmats, filenames):
    for m in range(len(filenames)):
        cmat = cmats[m, :, :, :].astype(np.uint8)
        if cmat.shape[0] == 1:
            ret_cmat = np.zeros((cmat.shape[1], cmat.shape[1]), dtype=np.uint8)
            for i in range(cmat.shape[1]):
                for j in range(cmat.shape[2]):
                    ret_cmat[i][j] = cmat[0][i][j]
        else:
            ret_cmat = np.zeros((cmat.shape[1], cmat.shape[1], 3), dtype=np.uint8)
            for i in range(cmat.shape[1]):
                for j in range(cmat.shape[2]):
                    ret_cmat[i][j] = np.array([cmat[0][i][j], cmat[1][i][j], cmat[2][i][j]])

        img = Image.fromarray(ret_cmat)
        img.save('./imgs/out/' + filenames[m])

# converts RGB color matrix to grayscale matrix
def rgb_to_grayscale_cmatrices(cmats):
    ret = np.zeros((cmats.shape[0], 1, cmats.shape[2], cmats.shape[3]), dtype=np.uint8)
    for m in range(cmats.shape[0]):
        cmat = cmats[m, :, :, :]
        for i in range(cmat.shape[1]):
            for j in range(cmat.shape[2]):
                ret[m][0][i][j] = 0.21*cmat[0][i][j] + 0.72*cmat[1][i][j] + 0.07*cmat[2][i][j]
    return ret


'''
Don't use -- needs reworking to work with volume dimensions properly
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



filenames = ['test1.png', 'test2.png']

cmats = imgs_to_cmatrices(filenames, n = 100)

gmats = rgb_to_grayscale_cmatrices(cmats)

cmatrices_to_imgs(gmats, filenames)