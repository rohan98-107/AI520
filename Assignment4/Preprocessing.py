from PIL import Image
import numpy as np

### IMAGE PRE-PROCESSING UTILITY FUNCTIONS ###

# converts image to matrix of color values
def imgs_to_cmatrices(filenames, n):
    rets = np.zeros((len(filenames), 3, n, n))
    for f in range(len(filenames)):
        img = Image.open('./imgs/' + filenames[f])
        img = img.convert('RGB')
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
