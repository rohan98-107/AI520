import numpy as np
import os
from Preprocessing import *

def NN_Model(grayscale_dataset, true_imgs, training_repeats = 1, learning_rate = 0.0025):
    # use Zhang et. al model architecture with no pooling layers, only conv w/ relu and upsampling
    m, _, img_size, img_size = grayscale_dataset.shape
    print(m)
    print(img_size)
    print(grayscale_dataset.shape)
    flattened = [ grayscale_dataset[i,:,:,:].reshape((img_size**2,1)) for i in range(m)]
    flattened_true = [true_imgs[i,:,:,:].reshape((img_size**2 * 3,1)) for i in range(m)]
    layer_1_weights = 2*np.random.random_sample((img_size**2 * 3, img_size**2))-2
    layer_2 = np.zeros(img_size**2 * 3)
    layer_2_weights = 2*np.random.random_sample((img_size**2 * 3, img_size**2 * 3))-2
    output = np.zeros(img_size**2 * 3)

    for i in range(training_repeats):
        for j in range(len(flattened)):
            print("training repeat: {} image: {}".format(i+1, j+1))
            layer_2 = np.maximum(0,np.matmul(layer_1_weights, flattened[i]))
            print(layer_2.shape)
            output = np.maximum(0,np.matmul(layer_2_weights, layer_2))
            print(output.shape)

            output_losses  = 2*(abs(output- flattened_true[i]))
            layer_2_gradients = np.zeros((img_size**2 * 3, img_size**2 * 3))
            layer_1_gradients = np.zeros((img_size**2 * 3, img_size**2 ))
            print("done feed forward")
            # compute gradient for weight from node at layer2[m] to output[k]
            for k in range(img_size**2 * 3):
                for m in range(img_size**2 * 3):
                    layer_2_gradients[k,m] = 2*(abs(output[k]- flattened_true[i][k])) * layer_2[m]
            print("done compute layer 2 gradient")
            # compute gradient for weight from node at layer1[l] to layer2[m]
            for m in range(img_size**2*3):
                for l in range(img_size**2):
                    if m ==0 and l == 0:
                        print(layer_2_weights[:,m].reshape((1,img_size**2*3)).shape)
                        print((2*(abs(output- flattened_true[i]))).shape)
                    layer_1_gradients[m,l] = np.matmul(layer_2_weights[:,m].reshape((1,img_size**2*3)), 2*(abs(output- flattened_true[i])))* flattened[i][l]


            print("done compute layer 1 gradient")

            layer_2_weights -= learning_rate * layer_2_gradients
            layer_1_weights -= learning_rate * layer_1_gradients

    return layer_1_weights, layer_2_weights


def grayscale_to_color_image(grayscale,filename,layer_1_weights,layer_2_weights):
    _, channels, img_size, img_size = grayscale.shape
    flattened = grayscale.reshape((1,img_size**2 * channels))
    layer_2 = np.maximum(0,np.matmul(flattened,layer_1_weights))
    output = np.minimum(np.maximum(0,np.matmul(layer_2, layer_2_weights)),255)
    output_reshaped = output.reshape((3,img_size,img_size))
    out_image = np.zeros((img_size,img_size,3))
    for i in range(img_size):
        for j in range(img_size):
            out_image[i][j][0] = output_reshaped[0][i][j]
            out_image[i][j][1] = output_reshaped[1][i][j]
            out_image[i][j][2] = output_reshaped[2][i][j]
    img = Image.fromarray(out_image, 'RGB')
    img.save(filename + '.png')

n = 20
image_files = ["animals/" + filename for filename in os.listdir("imgs/animals/")]
cmats = imgs_to_cmatrices(image_files,n)
gmats = rgb_to_grayscale_cmatrices(cmats)
print(cmats[0].shape)

layer1, layer2 = NN_Model(gmats[:-1,:,:,:],cmats[:-1,:,:,:])
f = file("tmp.bin","wb")
np.save(f,layer1)
np.save(f,layer2)
f.close()

f = file("tmp.bin","rb")
layer1 = np.load(f)
layer2 = np.load(f)
f.close()
grayscale_to_color_image(gmats[-1,:,:,:], "test", layer1, layer2)
