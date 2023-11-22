
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from django.shortcuts import get_object_or_404, render, redirect, get_object_or_404
from django.urls import reverse
import base64
import os
import errno
import cv2
import cv2 as cv
import time
import re
import noisereduce as nr
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io.wavfile as wav
import librosa
from pydub import AudioSegment
import speech_recognition as sr
from scipy.io import wavfile
import torch
import torchaudio
from pyAudioAnalysis import audioBasicIO, ShortTermFeatures
import math
import csv
from glob import glob
import librosa
import librosa.display
from itertools import cycle
import numpy as np
import pylab as plt
from scipy.fftpack import dct
from django.contrib import messages
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adagrad
from django.contrib.auth.decorators import login_required
from .models import Usuario, Sesion
from .forms import SesionFormHuella,MiFormularioSimple, SesionForm3, SesionFormVoz

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# Datos biométricos
   
def huella(request,usuario_id):
     if request.method == "GET":
         inquilino = get_object_or_404(Usuario,pk=usuario_id)
         form= SesionFormHuella(instance=inquilino)
         return render(request, 'sistemabio/huella.html', 
                       {  'inquilino':inquilino,
                          "form": form
                        })
     else:
          try:
               form = SesionFormHuella(request.POST)
               print("formulario", form.is_valid())
               new_huella = form.save(commit=False)
          except ValueError:
               messages.error(request, "Error no se creo el registro de huella.")
               return render(request, 'sistemabio/huella.html', 
                              { 'inquilino': inquilino,"form":  form , 
                              "error": "Error creando el registro de huella."})
     
     # title='huella'
     # return render (request,'sistemabio/vozjj.html',{
     #      'mytitle':title
     # })

def accediste(request):
    title='accediste'
    return render (request,'sistemabio/accediste.html',{
         'mytitle':title
    })

def huella_usuario(request):
    title='huella_usuario'
    return render (request,'sistemabio/huella_usuario.html',{
         'mytitle':title
    })




