import numpy as np
from mnist import MNIST
from ANN_prova_v5 import ANN_for_MNIST as ANN
from ANN_prova_v5 import Trainer_mnist as Trainer
from ANN_prova_v5 import normalization_img

def main():
    # Create instance of the model with some pre-trained model :)
    infile = "prova_salva_pesi.npz"
    Modello = ANN(load_file = infile)

    # Uploading and normalize images and labels...
    samples_path = ".\samples"
    mndata = MNIST(samples_path)

    imgs, labels = mndata.load_training()
    t_imgs, t_labels = mndata.load_testing()
    
    norm_imgs = normalization_img(imgs)
    norm_t_imgs = normalization_img(t_imgs)

    print(Modello.test_accuracy(norm_imgs, labels))
    print(Modello.test_accuracy(norm_t_imgs, t_labels))



if __name__ == "__main__":
    main()