# numpy == 1.14.3
# python == 3.6.5


import numpy as np
import h5py
from random import randint


np.random.seed(660943815)


global EPS 
EPS = 1e-10


def cross_entropy(yhat, y_true):
    loss = np.sum(-np.log(yhat + EPS) * create_target_vec(y_true))
    return loss


def softmax_function(z):
    ZZ = np.exp(z)/(np.sum(np.exp(z)) + EPS)
    return ZZ


def relu(h, derive=False):
    if derive:
        return 1. * (h > 0)
    else:
        return h * (h > 0)
    

def create_target_matrix(trues):
    T = np.zeros((len(trues), 10))
    for i, y in enumerate(trues):
        T[i,y] = 1
    return T


def create_target_vec(y):
    T = np.zeros(10)
    T[y] = 1
    return T


def accuracy_score(preds, true):
    return float(np.sum(preds == true)) / len(preds)


class Single_Layer_CNN(object):
    def __init__(self, input_dim, n_classes, filter_size, n_filters):

        tensor_dim = input_dim - filter_size + 1
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.tensor_dim = tensor_dim
        self.n_classes = n_classes
        self.input_dim = input_dim
        
        self.k = np.random.randn(filter_size, filter_size, n_filters) / np.sqrt(input_dim**2)
        
        self.weights = {'W':np.random.randn(n_classes, tensor_dim, tensor_dim, n_filters) / np.sqrt(input_dim**2),
                        'b1':np.random.randn(n_classes) / np.sqrt(input_dim**2),
                       }
        
        self.outputs = {'H': np.empty((tensor_dim, tensor_dim, n_filters)),
                        'Z': np.empty((tensor_dim, tensor_dim, n_filters)),
                        'U': np.empty(n_classes),
                        'soft_out': np.empty(n_classes)
                       }
                             
        
    def single_convolution(self, X, W, output_dim):

        output = np.zeros((output_dim, output_dim))
        for i in range(output_dim):
            for j in range(output_dim):
                output[i, j] = np.sum(X[i:i+len(W), j:j+len(W)] * W) 
            
        return output


    def convolution(self, X, k, derive=False):
        """
        Takes a input X as 3 dim tensor and filter k of 
        weights and returns the convolution output tensor
        """

        n_filters = k.shape[2]
        output_dim = len(X) - len(k) + 1
        output_tensor = np.empty((output_dim, output_dim, n_filters))

        # for each weight perform a single convolution 
        for i in range(n_filters):
            output_tensor[:, :, i] = self.single_convolution(X, k[:,:,i], output_dim)

        return output_tensor
    

    def forward_prop(self, X):
    
        Z = self.convolution(X, self.k)
        self.outputs['Z'] = Z

        H = relu(Z)
        self.outputs['H'] = H
        
        U = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            U[i] =  np.sum(self.weights['W'][i,:,:,:] * H)
        self.outputs['U'] = U
        
        soft_out = softmax_function(U)
        self.outputs['soft_out'] = soft_out
        
        return soft_out
    
    def predict(self, X):
        
        output = np.empty(len(X))
        for i in range(len(X)):
            output[i] = np.argmax(self.forward_prop(X[i,:,:]))

        return output
    
    def fit(self, X, y, n_epochs, LR, verbose):
        training_loss = []
        training_acc = []

        for epoch in range(n_epochs):
            print("Epoch = {}".format(epoch))
                
            #shuffle dataset
            idx = np.random.permutation(X.shape[0])
            X = X[idx, :, :]
            y = y[idx]

            # standard SGD
            batch_losses = []
            batch_acc = []
            for i in range(X.shape[0]):
                
                if (epoch > 40000):
                    LR = 1e-4
                if (epoch > 50000):
                    LR = 1e-5
                
                x_sample = X[i, :, :]
                y_sample = y[i]
                
                yhat = self.forward_prop(x_sample)
                
                # only check loss every 100
                batch_losses.append(cross_entropy(yhat, y_sample))
                if np.argmax(yhat) == y_sample:
                    batch_acc.append(1)
                else:
                    batch_acc.append(0)
                
                # perform network update
                self.backprop(x_sample, y_sample, yhat, LR)

                
            # training loss
            tmp_loss = np.mean(batch_losses)
            training_loss.append(tmp_loss)
            tmp_acc = np.mean(batch_acc)
            training_acc.append(tmp_acc)
            
            print("Training Loss = {}".format(tmp_loss))
            print("Training Acc = {}".format(tmp_acc))
            if verbose:
                print("Norm of weights = {}".format(np.linalg.norm(self.weights['W'])))
                print("Norm of weights = {}".format(np.linalg.norm(self.k)))
        
    
    def backprop(self, X, y, yhat, LR):
        clip = 0.5
        e_y = create_target_vec(y)
    
        delta1 = self.outputs['soft_out'] - e_y
        self.weights['b1'] += -LR * delta1.clip(-clip, clip)
        
        # update FC
        for i in range(self.n_classes):
            dp_dW = delta1[i] * self.outputs['H']
            self.weights['W'][i] += -LR * dp_dW.clip(-clip, clip)
        
        delta2 = np.zeros(self.weights['W'].shape)
        for k in range(self.n_classes):
            delta2[k,:,:,:] += (delta1[k] * self.weights['W'][k,:,:,:])
            
        delta2 = np.sum(delta2, axis=0)
            
        deltas = np.empty((self.tensor_dim, self.tensor_dim, self.n_filters))
        for p in range(self.n_filters):
            d_relu = relu(self.outputs['Z'][:,:,p], derive=True)
            deltas[:,:,p] = d_relu * delta2[:,:,p]

        dp_dk = self.convolution(X, deltas)

        for p in range(self.n_filters):
            self.k[:,:,p] += -LR * dp_dk[:,:,p].clip(-clip, clip)


def main():
	MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
	x_full = np.float32(MNIST_data['x_train'][:] )
	y_full = np.int32(np.array(MNIST_data['y_train'][:,0]))
	x_test = np.float32( MNIST_data['x_test'][:] )
	y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
	MNIST_data.close()

	img_dim = 28

	x_full_tensor = np.zeros((x_full.shape[0], 28, 28))

	for i, vec in enumerate(x_full):
	    for idx, j in enumerate(range(0, len(vec), img_dim)):
	        x_full_tensor[i, idx, :] = vec[j:j+28]
	        
	x_test_tensor = np.zeros((x_test.shape[0], 28, 28))

	for i, vec in enumerate(x_test):
	    for idx, j in enumerate(range(0, len(vec), img_dim)):
	        x_test_tensor[i, idx, :] = vec[j:j+28]

	model = Single_Layer_CNN(input_dim=28, n_classes=10, filter_size=5, n_filters=8)
	model.fit(x_full_tensor, y_full, n_epochs=1, LR=1e-3, verbose=False)

	test_preds = model.predict(x_test_tensor)
	print(accuracy_score(test_preds, y_test))


if __name__ == '__main__':
	main()