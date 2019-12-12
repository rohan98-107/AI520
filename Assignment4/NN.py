import numpy as np

def NN_Model(grayscale_dataset, true_imgs, training_repeats = 1, learning_rate = 0.0025):
    # use Zhang et. al model architecture with no pooling layers, only conv w/ relu and upsampling
    m, channels, img_size, img_size = grayscale_dataset.shape

    flattened = [ grayscale_dataset[i,:,:,:].reshape((1,img_size**2 * channels)) for i in range(m)]
    flattened_true = [true_imgs[i,:,:,:].reshape((1,img_size**2 * channels)) for i in range(m)]
    layer_1_weights = 2*np.random.random_sample((img_size**2 * channels, img_size**2 * channels))-2
    layer_2 = np.zeros(img_size**2 * channels)
    layer_2_weights = 2*np.random.random_sample((img_size**2 * channels, img_size**2 * channels))-2
    output = np.zeros(img_size**2 * channels)

    for i in range(training_repeats):
        for j in range(len(flattened)):
            layer_2 = np.maximum(0,np.matmul(flattened[i],layer_1_weights))
            output = np.maximum(0,np.matmul(layer_2, layer_2_weights))

            output_losses  = 2*(abs(output- flattened_true[i]))
            layer_2_gradients = np.zeros((img_size**2 * channels, img_size**2 * channels))
            layer_1_gradients = np.zeros((img_size**2 * channels, img_size**2 * channels))

            # compute gradient for weight from node at layer2[m] to output[k]
            for k in range(img_size**2 * channels):
                for m in range(img_size**2 * channels):
                    layer_2_gradients[k,m] = 2*(abs(output[k]- flattened_true[i][k])) * layer2[m]

            # compute gradient for weight from node at layer1[l] to layer2[m]
            for m in range(img_size**2 * channels):
                for l in range(img_size**2 * channels):
                    layer_1_gradients[l,m] = 0
                    for y in range(img_size**2 * channels):
                        layer_1_gradients[k,m] += 2*(abs(output[y]- flattened_true[i][y]))*

                    * layer1[m] * \
                            max(0,np.dot(layer_1_weights[k,:], layer_1))

            layer_2_weights -= learning_rate * layer_2_gradients
            layer_1_weights -= learning_rate * layer_1_gradients


dataset = np.random.randint(4,size= (1,1,2,2))
true_imgs = np.random.randint(4,size= (1,1,2,2))

NN_Model(dataset,true_imgs)
