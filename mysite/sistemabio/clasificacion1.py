#CLASIFICACIÓN DE AUDIO CON PYTORCH
import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio


class UrbanSoundDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        # archivo de anotaciones
        self.annotations = pd.read_csv(annotations_file)
        # directorio de audio
        self.audio_dir = audio_dir
        # gpu o cpu
        self.device = device
        # transformacion
        self.transformation = transformation.to(self.device)
        # frecuencia de muestreo
        self.target_sample_rate = target_sample_rate
        # numero de muestras
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # obtener path de audio
        audio_sample_path = self._get_audio_sample_path(index)
        # obtener label
        label = self._get_audio_sample_label(index)
        # cargar audio
        signal, sr = torchaudio.load(audio_sample_path)
        # definir la señal de entrada
        signal = signal.to(self.device)
        # aplicar remuestreo si es necesario (resampling)
        signal = self._resample_if_necessary(signal, sr)
        # convertir a señal monofonica si es necesario (mixing down)
        signal = self._mix_down_if_necessary(signal)
        # cortar si es necesario (cutting)
        signal = self._cut_if_necessary(signal)
        # ajustar si es necesario (padding)
        signal = self._right_pad_if_necessary(signal)
        # aplicar transformacion
        signal = self.transformation(signal)

        return signal, label

    def _cut_if_necessary(self, signal):
        """
        Corta la señal si es necesario, para ajustar el tamaño de la señal
        """        
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        """
        Ajusta la señal si es necesario, para ajustar el tamaño de la señal
        """
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        """
        Aplica el remuestreo si es necesario
        """

        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        """
        Convertir a señal monofonica si es necesario
        """
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        """
        Obtiene el path de audio
        """

        fold = f"fold{self.annotations.iloc[index, 4]}"   #el 5 indica el folder "fold" al que pertenece el audio
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[
            index, 0])
        return path

    def _get_audio_sample_label(self, index):
        """
        Obtiene el label del audio
        """

        return self.annotations.iloc[index, 5]


if __name__ == "__main__":
    # path de las anotaciones
    ANNOTATIONS_FILE = "C:/Users/yobis/Desktop/sistemabio/mysite/sistemabio/static/inquilinos/PruebaProyecto.csv"
        # directorio de audio
    AUDIO_DIR = "C:/Users/yobis/Desktop/sistemabio/mysite/sistemabio/static/voz"
    # frecuencia de muestreo
    SAMPLE_RATE = 22050
    # numero de muestras
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        # gpu disponible
        device = "cuda"
    else:
        # gpu no disponible
        device = "cpu"
    print(f"Using device {device}")


    # extraer el mel spectrogram de la señal
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        # frecuencia de muestreo
        sample_rate=SAMPLE_RATE,
        # tamaño de la FFT
        n_fft=1024,
        # tamaño del desplazamiento de la fft, normalemente fft/2
        hop_length=512,
        # numero de coeficientes del filtro mel
        n_mels=64
    )

    #definir el dataset
    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    print(f"El dataset tiene {len(usd)} archivos de audio.")
    signal, label = usd[0]




