import numpy as np
from mnist import MNIST
from ANN_prova_v5 import ANN_for_MNIST as ANN
from ANN_prova_v5 import Trainer_mnist as Trainer
from ANN_prova_v5 import normalization_img

def main():
    # Create instance of the model
    Modello = ANN()

    # Uploading and normalize images and labels...
    samples_path = ".\samples"
    mndata = MNIST(samples_path)
    images, labels = mndata.load_training()
    norm_images = normalization_img(images)

    # Creating instance of trainer and training phase
    Allenatore = Trainer(norm_images, labels, 0.01, 10, Modello)
    Allenatore.online_mode_train_e(norm_images, labels)

    # Saving weights 
    Modello.save_weights("prova_salva_pesi")





if __name__ == "__main__":
    main()
