
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

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

def perform_audio_prediction(audio_file_path):
    cnn = CNNNetwork()
    state_dict = torch.load("C:/Users/yobis/Desktop/sistemabio/mysite/cnnnet1PRUEBAUsers.pth")
    cnn.load_state_dict(state_dict)

    waveform, sample_rate = torchaudio.load(audio_file_path)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    mel_input = mel_spectrogram(waveform)
    mel_input = mel_input.unsqueeze(0)

    predicted_output = cnn(mel_input)
    predicted_index = predicted_output.argmax(1)
    predicted_class = class_mapping[predicted_index]

    print(f"Predicción para el archivo de audio: {audio_file_path}")
    print(f"Clase predicha: {predicted_class}")

    if predicted_class in class_mapping:
        return True
    else:
        return False

if __name__ == "__main__":
    audio_file_path = 'C:/Users/yobis/Desktop/sistemabio/mysite/sistemabio/static/newusuariov/audio_nuevo_usuario.wav'
    prediction_result = perform_audio_prediction(audio_file_path)
    print(f"Se encontró una predicción: {prediction_result}")
