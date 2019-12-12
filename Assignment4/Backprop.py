from CNN import *

def gradLoss(pred,true):
    return abs(2*(pred-true))

def backProp(filters,activations,true_img):

    init_shape = activations[-1].shape
    grad_losses = np.zeros(init_shape)

    for i in range(0,init_shape[2]):
        for j in range(0,init_shape[1]):
            for k in range(0,init_shape[0]):
                grad_losses[i,j,k] = gradLoss(activations[-1][i,j,k],true_img[i,j,k])

    print(filters[-1])
    print(grad_losses)

    filters[-1] = np.sum(np.multiply(grad_losses,np.maximum(0,np.sum(np.multiply(activations[-2],filters[-1])))))

    print(filters[-1])

    for t in range(len(activations)-2,-1,-1):

        if filters[t] is None:
            # handle upsampling
            continue
        else:
            filters[t] = np.sum(np.multiply(activations[t],np.maximum(0,np.sum(np.multiply(activations[t-1],filters[t])))))
