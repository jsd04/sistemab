import torch
import torchaudio

from clasificacion1 import UrbanSoundDataset
from cnn1 import CNNNetwork
from train1 import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES


class_mapping = [
    "fold",
    "maricruz",
    "mariela",
    "pamela"
]


def predict(model, input, target, class_mapping):
    """
    Predecir la clase de una secuencia de audio
    """


    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    
    # cargar el modelo
    cnn = CNNNetwork()
    state_dict = torch.load("cnnnet1PRUEBAUsers.pth")
    cnn.load_state_dict(state_dict)

   
    # instanciar la extración del mel spectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    # cargar el urban sound dataset

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            "cpu") # predicciones con CPU

    # obtencion un ejemplo de prueba del urban sound dataset para prediccion
    input, target = usd[0][0], usd[0][1] # [batch size, num_channels, fr, time]
    input.unsqueeze_(0)

    # realizar prediccion
    predicted, expected = predict(cnn, input, target,
                                  class_mapping)
    print(f"Prediccion: '{predicted}', Real: '{expected}'")





