#ENTRENAMIENTO 
import os
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from clasificacion1 import UrbanSoundDataset
from cnn1 import CNNNetwork


BATCH_SIZE = 44 #este es el número de ejemplos que se introducen en la red para que entrene de cada vez. Si el número es pequeño, significa que la red tiene en memoria poca cantidad de datos, y entrena más rápido
EPOCHS = 10 #cantidad de usuarios
LEARNING_RATE = 0.001


ANNOTATIONS_FILE = "C:/Users/yobis/Desktop/sistemabio/mysite/sistemabio/static/inquilinos/PruebaProyecto.csv"
        # directorio de audio
AUDIO_DIR = "C:/Users/yobis/Desktop/sistemabio/mysite/sistemabio/static/voz"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calcular la perdida y el gradiente
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # error de backpropagation y actualizacion de los pesos
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    
    # instanciar la extración del mel spectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    # Obtener la lista de archivos en el directorio de audio
    audio_files = [os.path.join(AUDIO_DIR, filename) for filename in os.listdir(AUDIO_DIR)]

    # Instanciar el dataset con la lista de archivos
    usd = UrbanSoundDataset(audio_files, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)

    # instanciar el dataset
    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    
    train_dataloader = create_data_loader(usd, BATCH_SIZE)

    # construcion del modelo y asignarlo a la GPU
    cnn = CNNNetwork().to(device)
    print(cnn)

    
    # inicializacion de la funcion de perdida + optimizador
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # entrenamiento del modelo
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # grabar el modelo
    torch.save(cnn.state_dict(), "cnnnet1PRUEBA.pth")
    print("red neuronal entrenada cnnnet1PRUEBA.pth")
