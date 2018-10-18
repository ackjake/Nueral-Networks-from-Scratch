# numpy==1.14.3
# Python 3.6.5


import h5py
import numpy as np
from random import randint


np.random.seed(660943815)


def load_data():
	MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
	x_full = np.float32(MNIST_data['x_train'][:] )
	y_full = np.int32(np.array(MNIST_data['y_train'][:,0]))
	x_test = np.float32( MNIST_data['x_test'][:] )
	y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
	MNIST_data.close()

	return x_full, y_full, x_test, y_test


def cross_entropy(yhat, y_true):
    loss = np.mean(np.sum(-np.log(yhat) * create_target_matrix(y_true), axis=1))
    return loss


def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
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


def accuracy_score(preds, true):
    return float(np.sum(preds == true)) / len(preds)


class Single_Layer_NN(object):
    def __init__(self, input_dim, n_units, n_classes):
        
        self.weights = {'W':np.random.randn(input_dim, n_units) / np.sqrt(n_units),
                        'b1':np.random.randn(n_units) / np.sqrt(n_units),
                        'C':np.random.randn(n_units, n_classes) / np.sqrt(n_units),
                        'b2':np.random.randn(n_classes) / np.sqrt(n_units)
                       }
        
        self.outputs = {'Z': np.empty(n_units),
                        'H': np.empty(n_units),
                        'soft_out': np.empty(n_classes)
                       }
        
    def forward_prop(self, X):

        Z = X.dot(self.weights['W']) + self.weights['b1']
        self.outputs['Z'] = Z      
        
        H = relu(Z)
        self.outputs['H'] = H
        
        U = H.dot(self.weights['C']) + self.weights['b2']
        soft_out = np.apply_along_axis(softmax_function, 1, U)
        self.outputs['soft_out'] = soft_out
        
        return soft_out
    
    def predict(self, X):
        yhat = self.forward_prop(X)
        return np.argmax(yhat, axis=1)
    
    def fit(self, X, y, x_val, y_val, n_epochs, batch_size):
        training_loss = []
        val_loss = []
        LR = 1e-3
        
        for epoch in range(n_epochs):
            #Learning rate schedule
            if (epoch > 25):
                LR = 1e-4
            if (epoch > 30):
                LR = 1e-5
                
            #shuffle dataset
            idx = np.random.permutation(len(X))
            X = X[idx, :]
            y = y[idx]

            # loop through batches within training set
            batch_losses = []
            for i in range(0, X.shape[0], batch_size):
                x_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                
                # forward propogate log loss/entropy per batch
                yhat = self.forward_prop(x_batch)
                batch_losses.append(cross_entropy(yhat, y_batch))
                
                # backprop and update weights
                self.backprop(x_batch, y_batch, LR)
                
            # training loss
            tmp_loss = np.mean(batch_losses)
            training_loss.append(tmp_loss)
            
            # validation loss + metrics
            val_yhat = self.forward_prop(x_val)
            val_tmp_loss = cross_entropy(val_yhat, y_val)
            val_acc = accuracy_score(np.argmax(val_yhat, axis=1), y_val)
            
            # report
            print("Epoch {} | Train Loss = {} | Val Loss = {} | Val Acc = {}".format(epoch+1,
                                                                                     np.round(tmp_loss, 4),
                                                                                     np.round(val_tmp_loss, 4),
                                                                                     np.round(val_acc, 4)))
            
    def backprop(self, X, y, LR):
                
        e_y = create_target_matrix(y)
        delta1 = self.outputs['soft_out'] - e_y
        delta2 = delta1.dot(self.weights['C'].T) * relu(self.outputs['Z'], derive=True)
        
        # gradients
        g_b2 = np.sum(delta1, axis=0)
        g_C = self.outputs['H'].T.dot(delta1)
        g_b1 = np.sum(delta2, axis=0)
        g_W = X.T.dot(delta2) 
        
        # update
        self.weights['b2'] += -LR * g_b2
        self.weights['C'] += -LR * g_C
        self.weights['b1'] += -LR * g_b1
        self.weights['W'] += -LR * g_W


def main():

	x_full, y_full, x_test, y_test = load_data()

	# create small 5000 observation validation set
	idx = 55000
	x_train = x_full[0:idx,:]
	y_train = y_full[0:idx]
	x_val = x_full[idx:,:]
	y_val = y_full[idx:]

	model = Single_Layer_NN(input_dim=28**2, n_classes=10, n_units=392)
	model.fit(x_train, y_train, x_val, y_val, n_epochs=40, batch_size=1000)

	test_acc = accuracy_score(model.predict(x_test), y_test)
	print("Test Accuracy = {}".format(test_acc))


if __name__== "__main__":
  main()