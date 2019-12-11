# Convolutional Neural Network from scratch
import numpy as np


def zeroPad(input, pad_val):  # make sure you only pad individual images
    return np.pad(input, ((0, 0), (0, pad_val), (0, pad_val)), mode='constant')


def convolve(input, filters, num_filters, spatial_extent, stride, padding, relu=False):
    '''
    MATLAB      NumPy
    ------------------
    cols        axis-1
    rows        axis-2
    dim3        axis-0
    '''

    # input volume
    d, W, H = input.shape
    K_check, d_check, F_check, F_check = filters.shape

    # hyperparameters
    K = num_filters
    F = spatial_extent
    S = stride
    P = padding

    # make sure everything matches up
    assert (d == d_check)
    assert (F == F_check)
    assert (K == K_check)

    # output volume dimensions
    W_prime = (W - F + 2 * P) // S + 1
    H_prime = (H - F + 2 * P) // S + 1
    d_prime = K

    # initialize output volume
    output = np.zeros((d_prime, W_prime, H_prime))

    # pad volume, prepare for convolution
    temp_volume = zeroPad(input, P)

    # perform convolution
    Z, X, Y = output.shape
    for x in range(X - P):
        for y in range(Y - P):
            # will assume: F = 5, W = H = 256, S = 1, P = 1 for now

            # capture every sub-volume of dimension (filter.shape)
            region = temp_volume[:, x * S:x * S + F, y * S:y * S + F]
            # then perform elementwise multiplication with that region and the filter
            for k in range(K):
                # populate one depth array of activation values
                output[k, x, y] = np.sum(np.multiply(region, filters[k, :, :, :]))

    if relu:
        output = np.maximum(output, 0)

    return output


def upSampling(matrix, factor):  # spatial upsampling

    return matrix.repeat(factor, axis=0).repeat(factor, axis=1)


def upSampleVolume(vol, factor):
    res = np.zeros((vol.shape[0], vol.shape[1] * factor, vol.shape[2] * factor))
    for i in range(vol.shape[0]):
        res[i, :, :] = upSampling(vol[i, :, :], factor)

    return res


def CNN_Model(grayscale_dataset):
    # use Zhang et. al model architecture with no pooling layers, only conv w/ relu and upsampling
    m, _, img_size, img_size = grayscale_dataset.shape

    # ************************************************

    FILTERS = []  # list of filter block matrices (list of 4d filter tensors)

    # initialize filters corresponding to layers

    # ----------------Layer 1-----------------------
    numfilters1 = 64
    filter1 = np.random.randint(-1, 2, size=(numfilters1, 1, 5, img_size))
    FILTERS.append(filter1)

    # ----------------Layer 2-----------------------
    numfilters2 = 128
    filter2 = np.random.randint(-1, 2, size=(numfilters2, numfilters1, 5, 5))
    FILTERS.append(filter2)

    # ----------------Layer 3-----------------------
    numfilters3 = 256
    filter3 = np.random.randint(-1, 2, size=(numfilters3, numfilters2, 5, 5))
    FILTERS.append(filter3)

    # ----------------Layer 4-----------------------
    numfilters4 = 512
    filter4 = np.random.randint(-1, 2, size=(numfilters4, numfilters3, 5, 5))
    FILTERS.append(filter4)

    # ----------------Layer 5-----------------------
    numfilters5 = 512
    filter5 = np.random.randint(-1, 2, size=(numfilters5, numfilters4, 5, 5))
    FILTERS.append(filter5)

    # ----------------Layer 6-----------------------
    numfilters6 = 256
    filter6 = np.random.randint(-1, 2, size=(numfilters6, numfilters5, 5, 5))
    FILTERS.append(filter6)

    # ************************************************

    # this is gonna be slow as FUCK

    # this is where we feed-forward
    for index in range(m):
        example = grayscale_dataset[index, :, :, :]

        # ************************************************

        # ----------------Layer 1-----------------------
        out1 = convolve(example, filter1, num_filters=numfilters1, spatial_extent=5, stride=1, padding=2, relu=True)

        # ----------------Layer 2-----------------------
        out2 = convolve(out1, filter2, num_filters=numfilters2, spatial_extent=5, stride=2, padding=2, relu=True)

        # ----------------Layer 3-----------------------
        out3 = convolve(out2, filter3, num_filters=numfilters3, spatial_extent=5, stride=2, padding=2, relu=True)

        # ----------------Layer 4-----------------------
        out4 = convolve(out3, filter4, num_filters=numfilters4, spatial_extent=5, stride=2, padding=2, relu=True)

        # ----------------Layer 5-----------------------
        out5 = convolve(out4, filter5, num_filters=numfilters5, spatial_extent=5, stride=1, padding=2)

        # ----------------Layer 5.5---------------------

        out5_upsampled = upSampleVolume(out5, 2)

        # ----------------Layer 6-----------------------
        out6 = convolve(out5_upsampled, filter6, num_filters=numfilters6, spatial_extent=5, stride=1, padding=2, relu=True)

        # ************************************************

    # then what we do with out6 I have no idea...

    return out6

def CNN_Model2(grayscale_dataset):
    # use Zhang et. al model architecture with no pooling layers, only conv w/ relu and upsampling
    m, _, img_size, img_size = grayscale_dataset.shape

    # ************************************************

    FILTERS = []  # list of filter block matrices (list of 4d filter tensors)
    F = 5 #spatial extent i.e. filter dimension

    # initialize filters corresponding to layers

    # ----------------Layer 1-----------------------
    numfilters1 = 15
    filter1 = np.random.randint(-1, 2, size=(numfilters1, 1, F, F))
    FILTERS.append(filter1)

    # ----------------Layer 2-----------------------
    numfilters2 = 30
    filter2 = np.random.randint(-1, 2, size=(numfilters2, numfilters1, F, F))
    FILTERS.append(filter2)

    # ----------------Layer 3-----------------------
    numfilters3 = 45
    filter3 = np.random.randint(-1, 2, size=(numfilters3, numfilters2, F, F))
    FILTERS.append(filter3)

    # ----------------Layer 4-----------------------
    numfilters4 = 15
    filter4 = np.random.randint(-1, 2, size=(numfilters4, numfilters3, F, F))
    FILTERS.append(filter4)

    # ----------------Layer 5-----------------------
    numfilters5 = 3
    filter5 = np.random.randint(-1, 2, size=(numfilters5, numfilters4, F, F))
    FILTERS.append(filter5)

    # ************************************************

    # this is gonna be slow as FUCK

    # this is where we feed-forward
    for index in range(m):
        example = grayscale_dataset[index, :, :, :]

        # ************************************************

        # ----------------Layer 1-----------------------
        out1 = convolve(example, filter1, num_filters=numfilters1, spatial_extent=F, stride=1, padding=2, relu=True)

        # ----------------Layer 2-----------------------
        out2 = convolve(out1, filter2, num_filters=numfilters2, spatial_extent=F, stride=2, padding=2, relu=True)

        # ----------------Layer 3-----------------------
        out3 = convolve(out2, filter3, num_filters=numfilters3, spatial_extent=F, stride=2, padding=2, relu=True)

        # ----------------Layer 3.5---------------------
        out3_upsampled = upSampleVolume(out3,2)

        # ----------------Layer 4-----------------------
        out4 = convolve(out3_upsampled, filter4, num_filters=numfilters4, stride=1, spatial_extent=F, padding=2, relu=True)

        # ----------------Layer 4.5---------------------
        out4_upsampled = upSampleVolume(out4,2)

        # ----------------Layer 5-----------------------
        out5 = convolve(out4_upsampled, filter5, num_filters=numfilters5, spatial_extent=F, stride=1, padding=2)

        # ************************************************

    # then what we do with out6 I have no idea...

    return out5

'''
data = np.random.randint(256, size=(1, 256, 256))
filters = np.random.randint(-1, 2, size=(64, 1, 5, 5))
out = convolve(data, filters, 64, 5, 2, 2, relu=True)
print(out.shape)
factor = 2
res = np.zeros((out.shape[0], out.shape[1] * factor, out.shape[2] * factor))
for i in range(out.shape[0]):
    res[i, :, :] = upSampling(out[i, :, :], factor)

print(res.shape)
'''
