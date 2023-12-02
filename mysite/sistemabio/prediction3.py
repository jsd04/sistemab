import torch
import torchaudio
from clasificacion1 import UrbanSoundDataset
from cnn1 import CNNNetwork
from train1 import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

class_mapping = [
    "fold1",
    "fold2",
    "fold3",
    "fold4"
]

def load_model(model_path):
    cnn = CNNNetwork()
    state_dict = torch.load(model_path)
    cnn.load_state_dict(state_dict)
    return cnn

def prepare_mel_spectrogram():
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    return mel_spectrogram

def perform_prediction(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

def main():
    cnn = load_model("C:/Users/yobis/Desktop/sistemabio/mysite/cnnnet1PRUEBAUsers.pth")
    mel_spectrogram = prepare_mel_spectrogram()

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            "cpu")  # predicciones con CPU

    input, target = usd[0][0], usd[0][1]  # [batch size, num_channels, fr, time]
    input.unsqueeze_(0)

    predicted, expected = perform_prediction(cnn, input, target, class_mapping)
    print(f"Prediccion: '{predicted}', Real: '{expected}'")

# Ejecutar main al importar el modulo
main()
