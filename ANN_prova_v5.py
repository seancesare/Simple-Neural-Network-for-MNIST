import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST


class ANN_for_MNIST():
    '''
    Artificial Neural Network class built for the hand written numbers \n
    classification task, using the MNIST dataset.    
    '''
    def __init__(self, layers: list = [784,20,15,10], load_file: str = None):
        '''
        Initialization of the 3 Weight matrices and biases
        '''
        if load_file:
            self.load_weights(load_file)
        else:
            self.layers = layers
            self.w1 = np.random.randn(layers[1], layers[0]) # 14x784    |
            self.w2 = np.random.randn(layers[2], layers[1]) # 7x14      | <-- matrices' dimensions
            self.w3 = np.random.randn(layers[3], layers[2]) # 10x7      |
            
            self.b1 = np.random.randn(layers[1])        # |  
            self.b2 = np.random.randn(layers[2])        # | <--- Biases
            self.b3 = np.random.randn(layers[3])        # |
        
    def relu(self, a: np.ndarray):
        '''
        Rectified Linear Unit (or ReLU in short). \n
        Outputs x for (x > 0), 0 for (x =< 0).
        '''
        return np.maximum(a, 0)        
    
    def relu_2(self, a):
        '''Old relu, to be cancelled soon'''
        return np.maximum(a, np.zeros(len(a)))

    def sigmoid(self, x):
        '''
        Sigmoidal function
        '''
        return 1 / (1 + np.exp(-x) )

    def relu_derivative(self, x):
        '''
        Computes derivative of the ReLU function.
        '''
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    
    def sigm_derivative(self, a):
        '''
        Computes the derivative of sigmoid function.
        '''
        sigma = self.sigmoid(a)
        return sigma * ( 1 - sigma )

    def quadratic_loss(self, y: int):
        '''
        Function for computing the quadratic loss for one image.
        y is the label needed for calculation of the error (int from 0 to 9)
        ''' 
        return 0.5 * (np.sum(np.square(self.output)) + 1 - 2*self.output[y])

    def entropy_loss(self, y: int):
        '''
        Function for computing the entropy loss for one image
        y is the label needed for calculation of the error (int from 0 to 9)
        '''
        return np.sum([-np.log(self.output[i]) if i==y else -np.log(1 - self.output[i]) for i in range(len(self.output))])

    def forward_step(self, image):
        '''
        Function for feedforwarding one image, from input to output.\n
        Returns numpy array of ten elements.
        '''
        self.x0 = np.array(image)

        # Input layer to first hidden layer with relu function 
        self.a1 = self.w1 @ self.x0 + self.b1
        self.x1 = self.relu(self.a1)

        # First hidden layer to second hidd. layer with relu func.
        self.a2 = self.w2 @ self.x1 + self.b2
        self.x2 = self.relu(self.a2)

        # Second hidden layer to output layers with sigmoidal func.
        self.a3 = self.w3 @ self.x2 + self.b3
        self.output = self.sigmoid(self.a3)

        return self.output  

    def backward_step_quadratic(self, label):
        '''
        Function for the calculation of the backward step and the delta errors\n
        in the case of quadratic loss function
        '''  
        # self.delta3 = [self.sigm_derivative(self.a3[i])*(self.output[i]) if i != y else for i in range(len(self.a3))]
        self.delta3 = np.zeros(len(self.output))

        # Computation of delta errors in output layer
        # for i in range(len(self.output)):
        #     # output[i] is basically the sigmoid of ai
        #     if i != label:
        #         self.delta3[i] = self.output[i] * self.output[i] * (1 - self.output[i])
        #     else:
        #         self.delta3[i] = self.sigm_derivative(self.a3[i]) * (self.output[i] - 1)
        
        # rewriting the above commented code in the below equivalent but more efficient way
        # Computation of delta errors in the output layer (derivative of the err. func. wrt activation)
        for i, output_i in enumerate(self.output):
            if i == label:
                self.delta3[i] = output_i * np.square(1 - output_i) 
            else:
                self.delta3[i] = output_i * output_i * (1 - output_i)
        
        # Hidden layer 2 delta errors
        self.delta2 = self.relu_derivative(self.a2) * (self.delta3 @ self.w3) 
        # * is the outer product, @ stands for matrix mult (row by col)
        
        # Hidd layer 1 delta errs
        self.delta1 = self.relu_derivative(self.a1) * (self.delta2 @ self.w2) 

    def backward_step_entropy(self, label):
        '''
        Function for the calculation of the backward step and the delta errors\n
        in the case of entropy loss function
        '''  
        # self.delta3 = [self.sigm_derivative(self.a3[i])*(self.output[i]) if i != y else for i in range(len(self.a3))]
        self.delta3 = np.zeros(self.layers[-1])

        # Computation of delta errors in output layer
        for i in range(self.layers[-1]):
            if i == label:
                self.delta3[i] = self.sigmoid(self.a3[i]) - 1   # riguarda la teoria, è una semplificazione della formula  
            else:
                self.delta3[i] = self.sigmoid(self.a3[i])
        
        # Hidden layer 2 delta errors
        self.delta2 = self.relu_derivative(self.a2) * (self.delta3 @ self.w3)  # * is the outer product, @ stands for matrix mult (row by col)
        
        # Hidd layer 1 delta errs
        self.delta1 = self.relu_derivative(self.a1) * (self.delta2 @ self.w2) 

    def grad_wrt_weights(self):
        # Gradient with respect to the weights' matrices and biases' matrices respectively
        self.grad3 = self.delta3.reshape(len(self.delta3),1) @ self.x2.reshape(1,len(self.x2))
        self.grad2 = self.delta2.reshape(len(self.delta2),1) @ self.x1.reshape(1,len(self.x1))
        self.grad1 = self.delta1.reshape(len(self.delta1),1) @ self.x0.reshape(1,len(self.x0))

        self.grad_b3 = self.delta3  # gradient for the bias is delta_i * x_j where x_j is 1, 
        self.grad_b2 = self.delta2  # so it's just the vector of the delta error itself
        self.grad_b1 = self.delta1
    
    def save_weights(self, outfile_name: str):
        '''Saves weights and biases into a file using numpy.savez function'''
        np.savez(outfile_name, 
            w1 = self.w1, b1 = self.b1,
            w2 = self.w2, b2 = self.b2,
            w3 = self.w3, b3 = self.b3)
        print(f"Weights have been saved into {outfile_name}.npz file!")
    
    def load_weights(self, outfile_name: str):
        '''Loads the weights and biases matrices and arrays'''
        with np.load(outfile_name) as npzfile:
            # loading of the weights
            self.w1 = npzfile['w1']
            self.b1 = npzfile['b1']
            self.w2 = npzfile['w2']
            self.b2 = npzfile['b2']
            self.w3 = npzfile['w3']
            self.b3 = npzfile['b3']

            # setting layers'numbers correctly
            layers = [28*28, len(self.b1), len(self.b2), len(self.b3)]
            self.layers = layers

    def save_model(self):
        '''Saves this instance of the model using pickle'''
        # not yet implemented, dunno if it will ever be lol.
        pass
    
    def test_accuracy(self, images, labels) -> float:
        '''It takes a list of images and then it simply computes the rate between the number of\n
        correct answers over all the answers.'''
        accuracy_counter = 0
        for idx, image in enumerate(images):
            self.forward_step(image)
            if np.argmax(self.output) == labels[idx]:
                accuracy_counter += 1
        precision = accuracy_counter/len(images)
        return precision

    def guess(self, norm_img, label: int = None):
        '''Takes one normalized image and label as input, returns the guess'''
        self.forward_step(norm_img)
        guessed_num = np.argmax(self.output)
        if guessed_num == label:
            print(f"Guessed number was {guessed_num}, the correct one is {label}... Looks I was right B-)")
        else:
            print(f"Guessed number was {guessed_num}, the correct one is {label}... Oopsie :)!")
        return guessed_num

                  
        
class Trainer_mnist():

    def __init__(self, images, labels, lr: float, epochs: int, Model: ANN_for_MNIST):
        self.epochs = epochs
        self.lr = lr
        self.Model = Model
        self.L = len(images)
    
    def online_mode_train_q(self, images, labels):
        "Takes as inputs the images and labels, computes the training with quadratic error"
        for epoch in range(self.epochs):
            
            accuracy_counter = 0

            for k in range(self.L):
                out = self.Model.forward_step(images[k])
                self.Model.backward_step_quadratic(labels[k])
                self.Model.grad_wrt_weights()
                
                self.Model.w1 = self.Model.w1 - self.lr * self.Model.grad1      # Updating the weights 
                self.Model.w2 = self.Model.w2 - self.lr * self.Model.grad2
                self.Model.w3 = self.Model.w3 - self.lr * self.Model.grad3

                self.Model.b1 = self.Model.b1 - self.lr * self.Model.grad_b1    # Updating the biases
                self.Model.b2 = self.Model.b2 - self.lr * self.Model.grad_b2
                self.Model.b3 = self.Model.b3 - self.lr * self.Model.grad_b3

                # printing out some infos...
                if not k % 3000:
                    print(f"Activation in last layer is: {self.Model.a3}")
                    print(f"Output is: {self.Model.output}")
                    print(f"Delta3 is: {self.Model.delta3}")
                    print(f"--- Label (correct value) is: {labels[k]},\n--- Predicted value is: {np.argmax(out)}")
                    print(f"The quadratic loss, on {k}-th iteration, epoch #{epoch} is: { self.Model.quadratic_loss(y=labels[k]):.4}")
                    print(f"The entropic loss, on {k}-th iteration, epoch #{epoch} is: {self.Model.entropy_loss(y=labels[k]):.4}\n")
            # printing accuracy and stuff...
            print(f"Weight matrix between penultimate and ultimate layer is: {self.Model.w3}")
            print(f"\n\n\n+++++++++ Accuracy (#number of correct guesses over every guess) for the {epoch}-th epoch was: {accuracy_counter*100/self.L :.6}% ++++++++\n\n\n\n")


    def online_mode_train_e(self, images, labels, print_infos = False):
        "Takes as inputs the images and labels, computes the training with quadratic error"
        for epoch in range(self.epochs):
            
            accuracy_counter = 0

            for k in range(self.L):
                out = self.Model.forward_step(images[k])
                self.Model.backward_step_entropy(labels[k])
                self.Model.grad_wrt_weights()
                
                self.Model.w1 = self.Model.w1 - self.lr * self.Model.grad1      # Updating the weights 
                self.Model.w2 = self.Model.w2 - self.lr * self.Model.grad2
                self.Model.w3 = self.Model.w3 - self.lr * self.Model.grad3

                self.Model.b1 = self.Model.b1 - self.lr * self.Model.grad_b1    # Updating the biases
                self.Model.b2 = self.Model.b2 - self.lr * self.Model.grad_b2
                self.Model.b3 = self.Model.b3 - self.lr * self.Model.grad_b3

                # increase the counter whenever the model is right
                if labels[k] == np.argmax(out):
                    accuracy_counter += 1 

                # printing out some infos...
                if print_infos:
                    if not k % 5000:
                        print(f"--- Label (correct value) is: {labels[k]},\n--- predicted value is: {np.argmax(out)}")
                        print(f"The quadratic loss, on {k}-th iteration, epoch #{epoch} is: { self.Model.quadratic_loss(y=labels[k]):.4}")
                        print(f"The entropic loss, on {k}-th iteration, epoch #{epoch} is: {self.Model.entropy_loss(y=labels[k]):.4}\n")
            # printing accuracy and stuff...
            print(f"\n\n\n+++++++++ Accuracy (#number of correct guesses over every guess) for the {epoch}-th epoch was: {accuracy_counter*100/self.L :.4}% ++++++++\n\n\n\n")

    def batch_mode_train(self):
        pass

    def mini_batch_train(self):
        pass


def display_image_with_matplotlib(image: list):
    '''
    Plots images[idx] into a gray-scale image
    '''
    image_array = np.array(image)           # transforms it in np array
    pixels = image_array.reshape((28,28))   # shape 784 dim vec into 28x28 matrix
    plt.imshow(pixels, cmap="gray")
    plt.show()

def print_ascii_mnist(images, labels, idx: int = 0, with_label: bool = True):
    '''
    It prints, from a list of 784 element, with integer values 
    in range(256), an image in ascii characters.\n
    It takes the index of the image in input.
    '''
    val_lum = ".,:;?%#@"  # string with 8 vals of luminosity ascii char

    for i in range(28):
        for j in range(28):
            val = images[idx][28*i+j]    # val of (i,j)-th pxl (0-255)
            lum_one_to_eight = val//32   # scaling of values in (0-7)
            val_in_ascii = val_lum[lum_one_to_eight] # picks the ascii val
            print(val_in_ascii, end="")
        print("")
    
    # Tells you the label associated with the idx-th image
    if with_label:
        print(f"This is number {labels[idx]} printed in ascii-style!\n({idx}-th image of the dataset)")


def normalization_img(images: list):
    '''
    Function for normalizing pixels values scaling them from [0;255] to [0;1]
    '''
    return [np.array(images[i])/255 for i in range(len(images))]

def main():

    # Uploading the mnist dataset and normalization of the images
    samples_path = ".\samples"
    mndata = MNIST(samples_path)
    images, labels = mndata.load_training()
    norm_images = normalization_img(images)

    # parameters for the trainer...
    lr = 0.005 
    epochs = 30
    layers = [28*28, 25, 15, 10]    
    # some good parameter for a simple yet effective model could be 25 and 15 for the 2 hidden layers, and 0.005 for lr

    Modello = ANN_for_MNIST(layers)
    Trainer = Trainer_mnist(norm_images, labels, lr, epochs, Modello)
    Trainer.online_mode_train_q(norm_images, labels)


if __name__ == "__main__":
    main()


   
