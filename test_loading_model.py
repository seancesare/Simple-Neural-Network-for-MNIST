import numpy as np
from mnist import MNIST
from ANN_prova_v5 import ANN_for_MNIST as ANN
from ANN_prova_v5 import Trainer_mnist as Trainer
from ANN_prova_v5 import normalization_img

def main():
    # Create instance of the model with some pre-trained model :)
    infile = "./saved_w/w_prova2.npz"
    Modello = ANN(load_file = infile)

    # Uploading and normalize images and labels...
    samples_path = ".\samples"
    mndata = MNIST(samples_path)

    imgs, labels = mndata.load_training()
    t_imgs, t_labels = mndata.load_testing()
    
    norm_imgs = normalization_img(imgs)
    norm_t_imgs = normalization_img(t_imgs)

    # testing accuracy for the pretrained models...
    print("Accuracy before training:")
    print(f"Accuracy in training dataset is: {Modello.test_accuracy(norm_imgs, labels)*100:.4}%")
    print(f"Accuracy in testing dataset is: {Modello.test_accuracy(norm_t_imgs, t_labels)*100:.4}%")

    # setting up parameters for trainer and stuff...
    lr, epochs = 0.002, 3
    Allenatore = Trainer(norm_imgs, labels, lr = lr, epochs = epochs, Model = Modello)
    Allenatore.online_mode_train_e(norm_imgs, labels)

    # testing again...
    print(f"Accuracy after training for {epochs} epochs with l.r. of {lr}:")
    print(f"Accuracy in training dataset is: {Modello.test_accuracy(norm_imgs, labels)*100:.4}%")
    print(f"Accuracy in testing dataset is: {Modello.test_accuracy(norm_t_imgs, t_labels)*100:.4}%")
    

    save_path = "./saved_w/w_prova3"
    Modello.save_weights(save_path)



if __name__ == "__main__":
    main()