import numpy as np
import os
from Preprocessing import *
from math import sqrt

def NN_Model(grayscale_dataset, true_imgs, training_repeats = 1, learning_rate = 0.0025, batch_size = 1, lamda = .01):
    # use Zhang et. al model architecture with no pooling layers, only conv w/ relu and upsampling
    m, _, img_size, img_size = grayscale_dataset.shape
    input_nodes = img_size**2
    output_nodes = img_size**2 * 3
    nodes_in_middle_layer = output_nodes * 3
    flattened = [ grayscale_dataset[i,:,:,:].reshape((input_nodes,1)) for i in range(m)]
    flattened_true = [true_imgs[i,:,:,:].reshape((output_nodes,1)) for i in range(m)]
    layer_1_weight_scale = 1
    layer_2_weight_scale = 1
    layer_3_weight_scale = 1
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
            # if (j+1) %10 == 0:
            #     print("training repeat: {} image: {}".format(i+1, j+1), flush=True)
            layer_2 = np.tanh(np.matmul(layer_1_weights, flattened[j]))
            layer_3 = np.tanh(np.matmul(layer_2_weights, layer_2))
            # sigmoid function, scaled
            output = (np.tanh((np.matmul(layer_3_weights, layer_3))) + 1) * (255/2)

            output_deltas  = 2*((output- flattened_true[j]))

            # derivative of sigmoid
            layer_3_intermediate = output_deltas * (1-((output * (2/255))-1)**2)
            layer_3_deltas = np.matmul(np.transpose(layer_3_weights),layer_3_intermediate)

            layer_2_intermediate = layer_3_deltas * (1-(layer_3)**2)
            layer_2_deltas = np.matmul(np.transpose(layer_2_weights), layer_2_intermediate)

            layer_1_intermediate = layer_2_deltas * (1-(layer_2)**2)
            # derivative of sigmoid

            layer_3_gradients += np.matmul(layer_3_intermediate, np.transpose(layer_3))
            layer_2_gradients += np.matmul(layer_2_intermediate, np.transpose(layer_2))
            layer_1_gradients += np.matmul(layer_1_intermediate, np.transpose(flattened[j]))


            #print("gradients zeroed?: ", layer_3_gradients.all() == 0, layer_2_gradients.all() == 0, layer_1_gradients.all() == 0)
            #print("intermediates zeroed?: ", layer_3_intermediate.all() == 0, layer_2_intermediate.all() == 0, layer_1_intermediate.all() == 0)
            #print("deltas zeroed?: ", layer_3_deltas.all() == 0, layer_2_deltas.all() == 0)

            if batch_size == 1 or (i* len(flattened) + j + 1) % batch_size == 0:
                # print("Adjusting weights")
                # layer_2_weights -= lamda * layer_2_weights/(i+1) + np.minimum(np.maximum(-.01 * batch_size * layer_2_weight_scale, learning_rate * layer_2_gradients * layer_2_weight_scale), .01 * batch_size * layer_2_weight_scale)
                # layer_1_weights -= lamda * layer_1_weights/(i+1) + np.minimum(np.maximum(-.01 * batch_size * layer_1_weight_scale, learning_rate ** 2 * layer_1_gradients * layer_1_weight_scale), .01 * batch_size * layer_1_weight_scale)
                # layer_2_weights -= np.minimum(np.maximum(-.01*batch_size,learning_rate * layer_2_gradients),.01*batch_size)
                # layer_1_weights -= np.minimum(np.maximum(-.01*batch_size,learning_rate * layer_1_gradients),.01*batch_size)
                layer_3_weights -= learning_rate * layer_3_gradients
                layer_2_weights -= learning_rate * layer_2_gradients
                layer_1_weights -= learning_rate * layer_1_gradients
                # layer_2_gradients = np.zeros((output_nodes, nodes_in_middle_layer))
                # layer_1_gradients = np.zeros((nodes_in_middle_layer, input_nodes ))
        train_set_error = 0
        for j in range(len(flattened)):
            input = flattened[j]
            layer_2 = np.tanh(np.matmul(layer_1_weights, input))
            layer_3 = np.tanh((np.matmul(layer_2_weights, layer_2)))
            output = (np.tanh((np.matmul(layer_3_weights, layer_3))) + 1) * (255/2)
            diffs = output- flattened_true[j]
            error = np.matmul(np.transpose(diffs),diffs)
            train_set_error += error[0,0]
        print()
        print("training repeat: {} average train set error: {}".format(i+1, train_set_error/len(flattened)))
        output_test(img_size,layer_1_weights,layer_2_weights, layer_3_weights)
    return layer_1_weights, layer_2_weights, layer_3_weights

def output_test(n,layer_1_weights,layer_2_weights, layer_3_weights):
    f_s = lambda x: 255 * sigmoid(x)

    image_files = ["test2/" + filename for filename in os.listdir("imgs/test2/") if filename[-3:] == "png" or filename[-3:] == "jpg"]
    cmats = imgs_to_cmatrices(image_files,n)
    gmats = rgb_to_grayscale_cmatrices(cmats)
    m,_, img_size, img_size = gmats.shape
    flattened_real = [cmats[i,:,:,:].reshape((n**2*3,1)) for i in range(m)]
    total_error = 0
    for i in range(m):
        flattened = gmats[i].reshape((img_size**2,1))
        layer_2 = np.tanh(np.matmul(layer_1_weights, flattened))
        layer_3 = np.tanh((np.matmul(layer_2_weights, layer_2)))

        # sigmoid function, scaled
        output = (np.tanh((np.matmul(layer_3_weights, layer_3))) + 1) * (255/2)
        diffs = output- flattened_real[i]
        error = np.matmul(np.transpose(diffs),diffs)
        total_error += error[0,0]

        output_reshaped = output.reshape((3,img_size,img_size))
        out_image = np.zeros((img_size,img_size,3))
        for j in range(img_size):
            for k in range(img_size):
                out_image[j][k][0] = output_reshaped[0][j][k]
                out_image[j][k][1] = output_reshaped[1][j][k]
                out_image[j][k][2] = output_reshaped[2][j][k]
        # print(out_image)
        img = Image.fromarray(out_image, 'RGB')
        img.save("imgs/tanh-"+str(n)+image_files[i] +"-colored"+ '.png')
        img = Image.fromarray(gmats[i].reshape(img_size,img_size), 'L')
        img.save("imgs/tanh-"+str(n)+image_files[i] +"-grayscale"+ '.png')
    print("average test set error: {}".format(total_error/m))
    print()
n = 16
image_files = ["train2/" + filename for filename in os.listdir("imgs/train2/") if filename[-3:] == "png" or filename[-3:] == "jpg"]
cmats = imgs_to_cmatrices(image_files,n)
gmats = rgb_to_grayscale_cmatrices(cmats)
# print(gmats[0])

layer1, layer2, layer3 = NN_Model(gmats,cmats,training_repeats=30, batch_size = 1)
f = open("tmp"+str(n)+".bin","wb")
np.save(f,layer1)
np.save(f,layer2)
np.save(f,layer3)
f.close()

f = open("tmp"+str(n)+".bin","rb")
layer1 = np.load(f)
layer2 = np.load(f)
layer3 = np.load(f)
f.close()
# grayscale_to_color_image(gmats[-30,:,:,:], image_files[-30][6:], layer1, layer2)

output_test(n,layer1,layer2, layer3)
