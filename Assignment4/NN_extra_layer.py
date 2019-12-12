import numpy as np
import os
from Preprocessing import *
from math import sqrt

def NN_Model(grayscale_dataset, true_imgs, training_repeats = 1, learning_rate = 0.000025, batch_size = 1):
    # use Zhang et. al model architecture with no pooling layers, only conv w/ relu and upsampling
    m, _, img_size, img_size = grayscale_dataset.shape
    input_nodes = img_size**2
    output_nodes = img_size**2 * 3
    nodes_in_middle_layer = output_nodes * 3
    flattened = [ grayscale_dataset[i,:,:,:].reshape((input_nodes,1)) for i in range(m)]
    flattened_true = [true_imgs[i,:,:,:].reshape((output_nodes,1)) for i in range(m)]
    layer_1_weight_scale = sqrt(6/(input_nodes+nodes_in_middle_layer))
    layer_2_weight_scale = sqrt(6/(nodes_in_middle_layer+output_nodes))
    layer_3_weight_scale = sqrt(6/(output_nodes))
    layer_1_weights = 2*layer_1_weight_scale*np.random.random_sample((nodes_in_middle_layer, input_nodes))-layer_1_weight_scale
    layer_2_weights = 2*layer_2_weight_scale*np.random.random_sample((nodes_in_middle_layer, nodes_in_middle_layer))-layer_2_weight_scale
    layer_3_weights = 2*layer_3_weight_scale*np.random.random_sample((output_nodes, nodes_in_middle_layer))-layer_3_weight_scale

    layer_1_gradients = np.zeros((nodes_in_middle_layer, input_nodes))
    layer_2_gradients = np.zeros((nodes_in_middle_layer, nodes_in_middle_layer))
    layer_3_gradients = np.zeros((output_nodes, nodes_in_middle_layer))
    # original_layer_2_gradients = np.zeros((output_nodes, nodes_in_middle_layer))
    # original_layer_1_gradients = np.zeros((nodes_in_middle_layer, input_nodes ))
    for i in range(training_repeats):
        for j in range(len(flattened)):
            layer_2 = np.maximum(0,np.matmul(layer_1_weights, flattened[i]))
            layer_3 = np.maximum(0,np.matmul(layer_2_weights, layer_2))
            output = np.maximum(0,np.matmul(layer_3_weights, layer_3))

            output_deltas  = 2*(abs(output- flattened_true[i]))
            # print("output deltas")
            layer_3_deltas = np.matmul(np.transpose(layer_3_weights),output_deltas)
            # print("layer 3 deltas")
            layer_2_deltas = np.matmul(np.transpose(layer_2_weights),layer_3_deltas)
            # print("layer 2 deltas")

            layer_3_gradients += np.matmul(output_deltas, np.transpose(layer_3))
            # print("layer 3 gradients")
            layer_2_gradients += np.matmul(layer_3_deltas, np.transpose(layer_2))
            # print("layer 2 gradients")
            layer_1_gradients += np.matmul(layer_2_deltas,np.transpose(flattened[i]))
            # print("layer 1 gradients")

            if batch_size == 1 or (i* len(flattened) + j + 1) % batch_size == 0:
                # print("Adjusting weights")
                layer_3_weights -= np.minimum(np.maximum(-.01*batch_size,learning_rate * layer_3_gradients),.01*batch_size)
                layer_2_weights -= np.minimum(np.maximum(-.01*batch_size,learning_rate * layer_2_gradients),.01*batch_size)
                layer_1_weights -= np.minimum(np.maximum(-.01*batch_size,learning_rate * layer_1_gradients),.01*batch_size)
                # layer_2_weights -= learning_rate * layer_2_gradients
                # layer_1_weights -= learning_rate**2 * layer_1_gradients
                layer_1_gradients = np.zeros((nodes_in_middle_layer, input_nodes))
                layer_2_gradients = np.zeros((nodes_in_middle_layer, nodes_in_middle_layer))
                layer_3_gradients = np.zeros((output_nodes, nodes_in_middle_layer))

            if (j+1) %10 == 0:
                print("training repeat: {} image: {}".format(i+1, j+1))
    return layer_1_weights, layer_2_weights, layer_3_weights


def grayscale_to_color_image(grayscale,filename,layer_1_weights,layer_2_weights):
    _, img_size, img_size = grayscale.shape
    flattened = grayscale.reshape((img_size**2,1))
    layer_2 = np.maximum(0,np.matmul(layer_1_weights, flattened))
    output = np.minimum(np.maximum(0,np.matmul(layer_2_weights, layer_2)),255)
    output_reshaped = output.reshape((3,img_size,img_size))
    out_image = np.zeros((img_size,img_size,3))
    for i in range(img_size):
        for j in range(img_size):
            out_image[i][j][0] = output_reshaped[0][i][j]
            out_image[i][j][1] = output_reshaped[1][i][j]
            out_image[i][j][2] = output_reshaped[2][i][j]
    print(out_image)
    img = Image.fromarray(out_image, 'RGB')
    img.save(filename +"-colored"+ '.png')
    img = Image.fromarray(grayscale.reshape(img_size,img_size), 'L')
    img.save(filename +"-grayscale"+ '.png')

def output_test(n,layer_1_weights,layer_2_weights,layer_3_weights):
    image_files = ["test2/" + filename for filename in os.listdir("imgs/test2/") if filename[-3:] == "png" or filename[-3:] == "jpg"]
    cmats = imgs_to_cmatrices(image_files,n)
    gmats = rgb_to_grayscale_cmatrices(cmats)
    m,_, img_size, img_size = gmats.shape

    for i in range(m):

        flattened = gmats[i].reshape((img_size**2,1))

        layer_2 = np.maximum(0,np.matmul(layer_1_weights, flattened))

        layer_3 = np.maximum(0,np.matmul(layer_2_weights, layer_2))

        output = np.minimum(np.maximum(0,np.matmul(layer_3_weights, layer_3)),255)

        output_reshaped = output.reshape((3,img_size,img_size))
        out_image = np.zeros((img_size,img_size,3))
        for j in range(img_size):
            for k in range(img_size):
                out_image[j][k][0] = output_reshaped[0][j][k]
                out_image[j][k][1] = output_reshaped[1][j][k]
                out_image[j][k][2] = output_reshaped[2][j][k]
        # print(out_image)
        img = Image.fromarray(out_image, 'RGB')
        img.save("imgs/extra-"+str(n)+image_files[i] +"-colored"+ '.png')
        img = Image.fromarray(gmats[i].reshape(img_size,img_size), 'L')
        img.save("imgs/extra-"+str(n)+image_files[i] +"-grayscale"+ '.png')

n = 64
image_files = ["train2/" + filename for filename in os.listdir("imgs/train2/") if filename[-3:] == "png" or filename[-3:] == "jpg"]
cmats = imgs_to_cmatrices(image_files,n)
gmats = rgb_to_grayscale_cmatrices(cmats)
# print(gmats[0])

layer1, layer2, layer3 = NN_Model(gmats,cmats,training_repeats=3, batch_size = 10)
f = open("tmp-extra-"+str(n)+".bin","wb")
np.save(f,layer1)
np.save(f,layer2)
np.save(f,layer3)
f.close()

f = open("tmp-extra-"+str(n)+".bin","rb")
layer1 = np.load(f)
layer2 = np.load(f)
layer3 = np.load(f)
f.close()
# grayscale_to_color_image(gmats[-30,:,:,:], image_files[-30][6:], layer1, layer2)

output_test(n,layer1,layer2,layer3)
