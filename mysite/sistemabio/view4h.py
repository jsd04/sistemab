
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
import time
from pyfingerprint.pyfingerprint import PyFingerprint
from pyfingerprint.pyfingerprint import FINGERPRINT_CHARBUFFER1
from pyfingerprint.pyfingerprint import FINGERPRINT_CHARBUFFER2

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

def registrar_huella():
    
     ## Enrolls new finger
     ##

     ## Tries to initialize the sensor
     try:
         f = PyFingerprint('/dev/ttyS0', 57600, 0xFFFFFFFF, 0x00000000)

         if ( f.verifyPassword() == False ):
             raise ValueError('La contraseña del sensor de huellas dactilares proporcionada es incorrecta!')

     except Exception as e:
         print('El sensor de huella no puede ser inicializado!')
         print('Exception message: ' + str(e))
         exit(1)

     ## Gets some sensor information
     print('Plantillas utilizadas actualmente: ' + str(f.getTemplateCount()) +'/'+ str(f.getStorageCapacity()))

     ## Tries to enroll new finger
     try:
         print('Esperando por una huella...')

         ## Wait that finger is read
         while ( f.readImage() == False ):
             pass

         ## Converts read image to characteristics and stores it in charbuffer 1
         f.convertImage(FINGERPRINT_CHARBUFFER1)

         ## Checks if finger is already enrolled
         result = f.searchTemplate()
         positionNumber = result[0]

         if ( positionNumber >= 0 ):
             print('La plantilla ya existe en la posición #' + str(positionNumber))
             exit(0)

         print('Remove finger...')
         time.sleep(2)

         print('Esperando por una huella otra vez...')

         ## Wait that finger is read again
         while ( f.readImage() == False ):
             pass

         ## Converts read image to characteristics and stores it in charbuffer 2
         f.convertImage(FINGERPRINT_CHARBUFFER2)

         ## Compares the charbuffers
         if ( f.compareCharacteristics() == 0 ):
             raise Exception('Las huellas no coinciden')

         ## Creates a template
         f.createTemplate()

         ## Saves template at new position number
         positionNumber = f.storeTemplate()
         print('Huella registrada exitosamente!')
         print('Nueva posición de plantilla #' + str(positionNumber))

     except Exception as e:
         print('Operación errorea!')
         print('Exception message: ' + str(e))
         exit(1)

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




