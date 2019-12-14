import numpy as np
import os
from Preprocessing import *
from math import sqrt

def NN_Model(grayscale_dataset, true_imgs, training_repeats = 1, learning_rate = 0.0000025, batch_size = 1, lamda = .01):
    m, _, img_size, img_size = grayscale_dataset.shape
    input_nodes = img_size**2
    output_nodes = img_size**2 * 3
    middle_layer_multiple  = 5
    nodes_in_middle_layer = output_nodes * middle_layer_multiple
    print("middle layer multiple = " + str(middle_layer_multiple))
    flattened = [ grayscale_dataset[i,:,:,:].reshape((input_nodes,1)) for i in range(m)]
    flattened_true = [true_imgs[i,:,:,:].reshape((output_nodes,1)) for i in range(m)]
    layer_1_weight_scale = .2
    layer_2_weight_scale = .2
    layer_1_weights = 2*layer_1_weight_scale*np.random.random_sample((nodes_in_middle_layer, input_nodes))-layer_1_weight_scale
    layer_2_weights = 2*layer_2_weight_scale*np.random.random_sample((output_nodes, nodes_in_middle_layer))-layer_2_weight_scale

    layer_2_gradients = np.zeros((output_nodes, nodes_in_middle_layer))
    layer_1_gradients = np.zeros((nodes_in_middle_layer, input_nodes))
    for i in range(training_repeats):
        print("training repeat: {}".format(i+1))
        for j in range(len(flattened)):
            if img_size >= 40:
                print("training repeat: {} image: {}".format(i+1, j+1))
            layer_2 = np.maximum(0,np.matmul(layer_1_weights, flattened[j]))

            output = np.minimum(np.maximum(0,np.matmul(layer_2_weights, layer_2)),255)

            output_deltas  = 2*((output- flattened_true[j]))
            layer_2_deltas = np.matmul(np.transpose(layer_2_weights),output_deltas)

            layer_2_gradients += np.matmul(output_deltas, np.transpose(layer_2))
            layer_1_gradients += np.matmul(layer_2_deltas,np.transpose(flattened[j]))

            if batch_size == 1 or (i* len(flattened) + j + 1) % batch_size == 0:
                # print("Adjusting weights")
                # layer_2_weights -= lamda * layer_2_weights + np.minimum(np.maximum(-.01*batch_size*layer_2_weight_scale,learning_rate * layer_2_gradients*layer_2_weight_scale/(i+1)**3),.01*batch_size*layer_2_weight_scale)
                # layer_1_weights -= lamda * layer_1_weights + np.minimum(np.maximum(-.01*batch_size*layer_1_weight_scale,learning_rate**2 * layer_1_gradients*layer_1_weight_scale/(i+1)**3),.01*batch_size*layer_1_weight_scale)
                layer_2_weights -= np.minimum(np.maximum(-.01*batch_size*layer_2_weight_scale,learning_rate * layer_2_gradients*layer_2_weight_scale),.01*batch_size*layer_2_weight_scale)
                layer_1_weights -= np.minimum(np.maximum(-.01*batch_size*layer_1_weight_scale,learning_rate**2 * layer_1_gradients*layer_1_weight_scale),.01*batch_size*layer_1_weight_scale)
                # layer_2_weights -= learning_rate * layer_2_gradients
                # layer_1_weights -= learning_rate**2 * layer_1_gradients
                layer_2_gradients = np.zeros((output_nodes, nodes_in_middle_layer))
                layer_1_gradients = np.zeros((nodes_in_middle_layer, input_nodes ))
        if (i+1) % 5 == 0:
            train_set_error = 0
            for j in range(len(flattened)):
                input = flattened[j]
                next_layer = np.maximum(0,np.matmul(layer_1_weights, input))
                output = np.minimum(np.maximum(0,np.matmul(layer_2_weights, next_layer)),255)
                diffs = output- flattened_true[j]
                error = np.matmul(np.transpose(diffs),diffs)
                train_set_error += error[0,0]
            print()
            print("training repeat: {} average train set error: {}".format(i+1, train_set_error/len(flattened)))
            output_test(img_size,layer_1_weights, layer_2_weights)
    return layer_1_weights, layer_2_weights

def output_test(n,layer_1_weights,layer_2_weights):
    image_files = ["test2/" + filename for filename in os.listdir("imgs/test2/") if filename[-3:] == "png" or filename[-3:] == "jpg"]
    cmats = imgs_to_cmatrices(image_files,n)
    gmats = rgb_to_grayscale_cmatrices(cmats)

    m,_, img_size, img_size = gmats.shape
    flattened_real = [cmats[i,:,:,:].reshape((n**2*3,1)) for i in range(m)]
    # print(m)
    total_error = 0
    for i in range(m):
        # print(image_files[i])
        flattened = gmats[i].reshape((img_size**2,1))
        layer_2 = np.maximum(0,np.matmul(layer_1_weights, flattened))
        output = np.minimum(np.maximum(0,np.matmul(layer_2_weights, layer_2)),255)
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
        img.save("imgs/"+str(n)+image_files[i] +"-colored"+ '.png')
        img = Image.fromarray(gmats[i].reshape(img_size,img_size), 'L')
        img.save("imgs/"+str(n)+image_files[i] +"-grayscale"+ '.png')

    print("average test set error: {}".format(total_error/m))
    print()

n = 16
image_files = ["train2/" + filename for filename in os.listdir("imgs/train2/") if filename[-3:] == "png" or filename[-3:] == "jpg"]
cmats = imgs_to_cmatrices(image_files,n)
gmats = rgb_to_grayscale_cmatrices(cmats)
# print(gmats[0])
print("dim " + str(n) + " NN")


layer1, layer2 = NN_Model(gmats,cmats,training_repeats=300, batch_size = 50)
f = open("tmp"+str(n)+".bin","wb")
np.save(f,layer1)
np.save(f,layer2)
f.close()

f = open("tmp"+str(n)+".bin","rb")
layer1 = np.load(f)
layer2 = np.load(f)
f.close()

output_test(n,layer1,layer2)