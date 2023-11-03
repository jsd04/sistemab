
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from django.http import HttpResponseRedirect,HttpResponse, Http404
from django.shortcuts import get_object_or_404, render, redirect, get_object_or_404
from django.urls import reverse
from django.views import generic
from django.utils import timezone
from django.template import loader
import datetime
import base64
import os
import errno
import cv2
import cv2 as cv
import time
import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import librosa
from pydub import AudioSegment
import speech_recognition as sr
from scipy.io import wavfile
import torch
import torchaudio
#creatiionform
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from django.contrib.auth import login, logout, authenticate
from django.db import IntegrityError
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.contrib import messages
from django.core.paginator import Paginator
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adagrad

from .models import Usuario, Sesion
from .forms import InquilinoForm, SesionForm, SesionFormUsuario,MiFormularioSimple, SesionForm3, SesionFormVoz

# Biométricos
@login_required
def new_biometricos(request):
      if request.method == "GET":
         form= SesionForm
         inquilinos = Usuario.objects.all()
         return render(request, 'sistemabio/inquilinos/new-biometricos.html', 
                       {"form": SesionForm,
                        'inquilinos': inquilinos
                        })   
      else:
         form= SesionForm
         inquilinos = Usuario.objects.all()
         return render(request, 'sistemabio/inquilinos/new-biometricos.html', 
                       {"form": SesionForm,
                        'inquilinos': inquilinos
                        })  

@login_required
def new_biometrico(request, usuario_id):
     if request.method == "GET":
        inquilino = get_object_or_404(Usuario,pk=usuario_id)
        form= SesionForm(instance=inquilino)
        return render(request, 'sistemabio/inquilinos/new-biometrico.html', 
                      {"form": SesionForm, 'inquilino':inquilino,
                       })   
     else:
        # print(request.POST)
        # return render(request, 'sistemabio/inquilinos/new-inquilino.html', {"form":  InquilinoForm})
        try:
            form = SesionForm(request.POST)
            new_biometricos = form.save(commit=False)
            new_biometricos.save()
            messages.success(request," El registro biométrico ha sido un éxito.")
            return redirect('/sistemabio/new_biometrico/')
        except ValueError:
            messages.error(request, "Error no se registro el biométrico.")
            return render(request, 'sistemabio/inquilinos/new-biometrico.html', 
                          {"form":  SesionForm,
                           "error": "Error registrando el biométrico."})
        

# Datos biométricos
def facial(request, usuario_id):
     if request.method == "GET":
         inquilino = get_object_or_404(Usuario,pk=usuario_id)
         form= SesionForm3(instance=inquilino)
         return render(request, 'sistemabio/facial.html', 
                       {  'inquilino':inquilino,
                          "form": form
                        })
     else:
          try:
               form = SesionForm3(request.POST)
               print("formulario", form.is_valid())
               print('form ', form['dato'].value())
               new_facial = form.save(commit=False)
               inquilino = get_object_or_404(Usuario,pk=usuario_id)
               print('id usuario: ', usuario_id)
               print('usuario: ',inquilino.nombre, inquilino.ap_paterno, inquilino.ap_materno ) 
               # sesion = get_object_or_404(Sesion,id_usuario_id=usuario_id)
               # print('id tipo sesion : ',sesion.id_tipo_sesion)
               # print('=======================')   
               # print('sesion dato', sesion.dato)
               personName =  str(usuario_id) + inquilino.nombre + inquilino.ap_paterno + inquilino.ap_materno 
               print("Nombre de personName es: ", personName)
               dataPath = 'C:/Users/yobis/Desktop/sistemabio/mysite/sistemabio/static/inquilinos' + '/' + personName
               personPath = dataPath + '/' + 'FACIAL' + personName
               print("Nombre de carpeta es: ", personPath)
               if not os.path.exists(personPath):
                    try:
                         os.mkdir(personPath, mode=0o755)
                         print('Carpeta creada: ',personPath)
                    except OSError as e:
                         if e.errno!=errno.EEXIST:
                              raise
               else :
                    print('el directorio ya existe')
               print('form dato: ', form['dato'].value())
               dato = form['dato'].value()
               dato_rep = str(dato).replace('data:image/jpeg;base64,', '')
               print('dato_rep: ', dato_rep)
               #División de la cadena
               datos_div = dato_rep.split(',')
               # Procesar cada elemento en un bucle
               i=0
               for dato_div in datos_div:
                    # Realiza alguna acción con el elemento, por ejemplo, imprimirlo
                    variable = datos_div[i]
                    print('*************///////////////////////////////********************')
                    print('cadena dividida: ',variable)
                    # variable_rep = variable.rstrip(variable[-1])
                    # print('cadena div_rep: ',variable_rep)
                    #primero codificamos de string/cadena/caracteres a bytes por que la función b64encode no recibe str como parámetro, sino bytes
                    dato_utf= variable.encode('utf-8')
                    print('imagen decode: ',dato_utf )
                    #decodificamos los bytes en base64
                    img_decode = base64.b64decode(dato_utf)
                    img_name= 'rostro_{}.jpg'.format(i)
                    img_file = open(personPath+'/'+img_name, 'wb')
                    img_file.write(img_decode)
                    i += 1
               # inicia la deteccion y recorte 
               faceClassif = cv2.CascadeClassifier('C:/Users/yobis/Desktop/sistemabiors/SistemaBiometricoJessi/mysite/sistemabio/static/haarcascades/haarcascade_frontalface_default.xml')
               captureList = os.listdir(personPath)
               print('lista de imagenes', captureList)
               image_array = []
               count = 0
               for filename in captureList:
                    imagepath = personPath+"/"+filename
                    print(imagepath)
                    image = cv2.imread(imagepath)
                    if image is None: continue
                    imageAux = cv2.imread(imagepath)
                    gray = cv2.cvtColor(image,  cv2.COLOR_BGR2GRAY)
                    faces = faceClassif.detectMultiScale(gray, 1.1, 5)
                    for (x, y, w, h) in faces:
                         cv2.rectangle(image, (x, y), (x + w, y + h), (128, 0, 255), 2)
                         cv2.rectangle(image, (10, 5), (450, 25), (255, 255, 255), 2)
                         rostro = imageAux[y:y + h, x:x + w]
                         rostro = cv2.resize(rostro, (224, 224), interpolation=cv2.INTER_CUBIC)
                         cv2.imwrite(personPath +'/'+ filename, rostro)
                         print('leyendo imagenerecorte')

                    imagepath = personPath+"/"+filename
                    print(imagepath)
                    image_file = open(personPath +'/'+filename, 'rb')
                    image = image_file.read()
                    
                    
               count += 1
               print(new_facial)
               new_facial.dato = image
               print(new_facial.dato)
               new_facial.completado = True
               print(new_facial.completado)
               new_facial.save()
               cnn_facial(personPath)
               print('si termino')
                    
               messages.success(request," El registro facial ha sido un éxito.")
               return redirect('/sistemabio/inquilinos/')
          except ValueError:
               messages.error(request, "Error no se creo el registro facial.")
               return render(request, 'sistemabio/facial.html', 
                              { 'inquilino': inquilino,"form":  form , 
                              "error": "Error creando el registro facial."})
          


def voz3(request, usuario_id):
    if request.method == "GET":
        inquilino = get_object_or_404(Usuario, pk=usuario_id)
        form = SesionFormVoz(instance=inquilino)
        return render(request, 'sistemabio/vozjj.html', {
            'inquilino': inquilino,
            "form": form
        })
    else:
        form = SesionFormVoz(request.POST)
        new_voz = form.save(commit=False)

        if form.is_valid():
            print("formulario", form.is_valid())
            dato = form['dato'].value()
            inquilino = get_object_or_404(Usuario, pk=usuario_id)
            personName = str(usuario_id) + inquilino.nombre + inquilino.ap_paterno + inquilino.ap_materno
            print("Nombre de personName es: ", personName)
            dataPath = 'C:/Users/yobis/Desktop/sistemabio/mysite/sistemabio/static/inquilinos/' + personName
            personPath = dataPath + '/' + 'VOZ' + personName
            print("Nombre de carpeta es: ", personPath)
            if not os.path.exists(personPath):
                try:
                    os.makedirs(personPath, mode=0o755)
                    print('Carpeta creada:', personPath)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
            else:
                print('el directorio ya existe')
            # Decodificar la cadena Base64
            datos_decodificados = base64.b64decode(dato)
            # Guardar los datos de audio en un archivo
            voz_name = 'audio_user_'+ str(usuario_id) +'.wav'
            print('voz_name: ',voz_name)
            # Ahora, tienes los datos de audio en su formato original en el archivo "audio_original.wav"
            with open(personPath+'/'+voz_name, 'wb') as audio_file:
                 audio_file.write(datos_decodificados)
            print('audio guardado')
            captureList = os.listdir(personPath)
            print('lista de voices', captureList)
            voz_array = []
            for filename in captureList:
                vozpath = personPath+"/"+filename
                print(vozpath)
                 # Cargar el archivo de audio en formato MP3 (o el formato que tengas)
                audio = AudioSegment.from_file(vozpath)
                # Ruta de salida para el archivo WAV
                output_wav_path = personPath + "/" + filename
                # Exportar el archivo a formato WAV
                audio.export(output_wav_path, format="wav")
                print(f"El archivo ha sido convertido a WAV y guardado en: {output_wav_path}")
                 # Leer el archivo WAV
                sample_rate, audio_data = wav.read(output_wav_path)
                print('si leee')
                # Convertir a formato mono si es estéreo
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                # Frase específica que deseas comparar
                frase_especifica = "prueba del 31 de octubre del 23"
                # Identificar la voz humana usando SpeechRecognition
                r = sr.Recognizer()
                with sr.AudioFile(vozpath) as source:
                    audio_text = r.record(source)
                    try:
                        recognized_text = r.recognize_google(audio_text, language='es-MX')
                        print(f"Texto reconocido del archivo {filename}: {recognized_text}")
                        # Comparar con la frase específica
                        if recognized_text.lower() == frase_especifica.lower():
                            print("Sí, es la frase correcta.")
                            # Eliminar ruido utilizando pydub
                            audio = AudioSegment.from_wav(vozpath)
                            # Aplicar reducción de ruido (ajusta el valor según tus necesidades)
                            audio = audio - 20
                            # Guardar el audio limpio en un nuevo archivo
                            audio_cleaned_path = os.path.join(personPath, 'cleaned_' + filename)
                            audio.export(audio_cleaned_path, format="wav")
                            print(f"Archivo de audio limpio guardado en: {audio_cleaned_path}")
                            # Leer el archivo limpio en formato binario
                            with open(audio_cleaned_path, "rb") as audio_file:
                                audio_bytes = audio_file.read()
                            # Codificar en base64
                            datos_codificados = base64.b64encode(audio_bytes)
                            new_voz.dato = datos_codificados
                            new_voz.completado = True
                            print(new_voz.completado)
                            new_voz.save()
                            print('Si se guardo el formulario')
                            messages.success(request, "El registro de voz ha sido un éxito.")
                            return redirect('/sistemabio/inquilinos/')
                        else:
                             messages.error(request, "Error: Revisa que la frase sea correcta.")
                             return render(request, 'sistemabio/vozjj.html', {
                                 'inquilino': inquilino,
                                 "form": form,
                                 "error": "Error creando el registro de voz."
                             })
                    except sr.UnknownValueError:
                        print(f"No se pudo reconocer la voz en el archivo {filename}.")

        else:
            messages.error(request, "Error: no se creó el registro de voz.")
            return render(request, 'sistemabio/vozjj.html', {
                'inquilino': inquilino,
                "form": form,
                "error": "Error creando el registro de voz."
            })


def voz2(request, usuario_id):
    if request.method == "POST":
        form = SesionFormVoz(request.POST)
        print('form ', form['id_tipo_sesion'].value())
        print('form ', form['dato'].value())
        if form.is_valid():
            form.save()  # Guarda el formulario
            messages.success(request, "El registro de voz ha sido un éxito.")
            return redirect('/sistemabio/inquilinos/')
        else:
            messages.error(request, "Error creando el registro de voz.")

    else:
        inquilino = get_object_or_404(Usuario, pk=usuario_id)
        form = SesionFormVoz(instance=inquilino)

    return render(request, 'sistemabio/vozjj.html', {
        'inquilino': inquilino,
        "form": form
    })
                   
def huella(request,usuario_id):
     if request.method == "GET":
         inquilino = get_object_or_404(Usuario,pk=usuario_id)
         form= SesionForm3(instance=inquilino)
         return render(request, 'sistemabio/voz.html', 
                       {  'inquilino':inquilino,
                          "form": form
                        })
     else:
          try:
               form = SesionForm3(request.POST)
               print("formulario", form.is_valid())
               new_huella = form.save(commit=False)
          except ValueError:
               messages.error(request, "Error no se creo el registro de huella.")
               return render(request, 'sistemabio/voz.html', 
                              { 'inquilino': inquilino,"form":  form , 
                              "error": "Error creando el registro de huella."})
     
     # title='huella'
     # return render (request,'sistemabio/vozjj.html',{
     #      'mytitle':title
     # })

def cnn_facial(personPath):

    # dirname = os.path.join(os.getcwd(), 'usuarioscop')
    # imgpath = dirname + os.sep
    # inquilino = get_object_or_404(Usuario, pk=usuario_id) 
    # personName =  str(usuario_id) + inquilino.nombre + inquilino.ap_paterno + inquilino.ap_materno 
    # dataPath = 'C:/Users/yobis/Desktop/sistemabio/mysite/sistemabio/static/inquilinos' + '/' + personName
    # personPath = dataPath + '/' + 'FACIAL' + personName
    time.sleep(30)
    captureList = os.listdir(personPath)
    images = []
    directories = []
    dircount = []
    prevRoot=''
    cant=0
    print("leyendo imagenes de ",personPath)
    for root, dirnames, filenames in os.walk(personPath):
        for filename in filenames:
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                cant=cant+1
                filepath = os.path.join(root, filename)
                image = plt.imread(filepath)
                images.append(image)
                b = "Leyendo..." + str(cant)
                print (b, end="\r")
                if prevRoot !=root:
                    print(root, cant)
                    prevRoot=root
                    directories.append(root)
                    dircount.append(cant)
                    cant=0
    dircount.append(cant)
    #Se crean las etiquetas
    labels=[]
    indice=0
    for cantidad in dircount:
        for i in range(cantidad):
            labels.append(indice)
        indice=indice+1
    print("Cantidad etiquetas creadas: ",len(labels))
    deportes=[]
    indice=0
    for directorio in directories:
        name = directorio.split(os.sep)
        print(indice , name[len(name)-1])
        deportes.append(name[len(name)-1])
        indice=indice+1
    y = np.array(labels)
    X = np.array(images, dtype=np.vectorize) #convierto de lista a numpy
    print("Find the unique numbers from the train labels")
    # Find the unique numbers from the train labels
    classes = np.unique(y)
    nClasses = len(classes)
    print('Total number of outputs : ', nClasses)
    print('Output classes : ', classes)
    #Creamos Set de entranamiento y test
    print("Creamos Set de entranamiento y test")
    train_X,test_X,train_Y,test_Y = train_test_split(X,y,test_size=0.2)
    print('Training data shape : ', train_X.shape, train_Y.shape)
    print('Testing data shape : ', test_X.shape, test_Y.shape)
    print(train_X.shape)
    print(test_X.shape)
    print( train_Y.shape)
    print(test_Y.shape)
    print("PRocesamos las imagenes")
    #PRocesamos las imagenes
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    train_X = train_X / 255.
    test_X = test_X / 255.
    print("Hacemos el hot-encoding")
    #Hacemos el hot-encoding
    # Change the labels from categorical to one-hot encoding
    train_Y_one_hot = to_categorical(train_Y)
    test_Y_one_hot = to_categorical(test_Y)
    print("Display the change for category label using one-hot encoding")
    # Display the change for category label using one-hot encoding
    print('Original label:', train_Y[0])
    print('After conversion to one-hot:', train_Y_one_hot[0])
    #Creamos el set de entrenamiento y validación 
    print("Creamos el set de entrenamiento y validación ")
    #Mezclar todo y crear los grupos de entrenamiento y testing
    train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
    print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)
    #Creamos el modelo CNN
    print("Creamos el modelo CNN")
    #declaramos variables con los parámetros de configuración de la red
    INIT_LR = 1e-3 # Valor inicial de learning rate. El valor 1e-3 corresponde con 0.001
    epochs = 6 # Cantidad de iteraciones completas al conjunto de imagenes de entrenamiento
    batch_size = 64 # cantidad de imágenes que se toman a la vez en memoria
    sport_model = Sequential()
    sport_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(224,224,3)))
    sport_model.add(LeakyReLU(alpha=0.1))
    sport_model.add(MaxPooling2D((2, 2),padding='same'))
    sport_model.add(Dropout(0.5))
    sport_model.add(Flatten())
    sport_model.add(Dense(32, activation='linear'))
    sport_model.add(LeakyReLU(alpha=0.1))
    sport_model.add(Dropout(0.5))
    sport_model.add(Dense(nClasses, activation='softmax'))
    sport_model.summary()
    # Definir la tasa de aprendizaje inicial y la decadencia
    INIT_LR = .001
    denom = 11
    decay = INIT_LR / denom
    decay = .000105
    INIT_LR = 0.00116
    # Crear el optimizador Adagrad con la tasa de aprendizaje y la decadencia
    optimizer = Adagrad(learning_rate=INIT_LR, initial_accumulator_value=decay)
    # Compilar el modelo
    sport_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adagrad(learning_rate=INIT_LR, initial_accumulator_value=decay),metrics=['accuracy'])
    #Entrenamos el modelo: aprende a clasificar imágenes
    print("Entrenamos el modelo: aprende a clasificar imágenes")
    # este paso puede tomar varios minutos, dependiendo de tu ordenador, cpu y memoria ram libre
    sport_train = sport_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
    # guardamos la red, para reutilizarla en el futuro, sin tener que volver a entrenar
    sport_model.save("facecop_mnist_jessi.h5py")
    #Evaluamos la red
    print("Evaluamos la red")
    test_eval = sport_model.evaluate(test_X, test_Y_one_hot, verbose=1)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])
    predicted_classes2 = sport_model.predict(test_X)
    predicted_classes=[]
    for predicted_sport in predicted_classes2:
        predicted_classes.append(predicted_sport.tolist().index(max(predicted_sport)))
    predicted_classes=np.array(predicted_classes)
    predicted_classes.shape, test_Y.shape
    #Aprendemos de los errores: qué mejorar? 
    correct = np.where(predicted_classes==test_Y)[0]
    print("Found %d correct labels" % len(correct))
    incorrect = np.where(predicted_classes!=test_Y)[0]
    print("Found %d incorrect labels" % len(incorrect))


def facial_usuario(request):
    if request.method == "GET":
        form = MiFormularioSimple()
        return render(request, 'sistemabio/facial_usuario.html', {
            "form": form
        })
    else:
        # form = MiFormularioSimple(request.POST)
        form = MiFormularioSimple(request.POST)
        # new_facial_usuario = form.save(commit=False)
        print('form : ', form['dato_simple'].value())
        if form.is_valid():
            personPath = 'C:/Users/yobis/Desktop/sistemabio/mysite/sistemabio/static/newusuario'
            personPath_2 = 'C:/Users/yobis/Desktop/sistemabio/mysite/sistemabio/static/inquilinos'
            dato_simple = form['dato_simple'].value()
            dato_rep = str(dato_simple).replace('data:image/jpeg;base64,', '')
            print('dato_rep: ', dato_rep)
            dato_utf= dato_rep.encode('utf-8')
            print('imagen decode: ',dato_utf )
            #decodificamos los bytes en base64
            img_decode = base64.b64decode(dato_utf)
            img_name= 'nuevo_usuario.jpg'
            img_file = open(personPath+'/'+img_name, 'wb')
            img_file.write(img_decode)
            imagepath = detec_recorte(personPath)
            
            #Aqui mismo hacer reconocimiento
            if(reconocimiento_facial(imagepath,personPath_2)):
                messages.success(request, "El reconocimiento facial ha sido un éxito.")
                return redirect('/sistemabio/accediste/')
            # print(new_facial_usuario.dato)
            # new_facial_usuario.save()# Guarda el formulario
            else:
                messages.error(request, "Error no se creo el registro facial.")
                return render(request, 'sistemabio/facial_usuario.html', 
                        { "form":  form, 
                        "error": "Error creando el registro facial."})

def detec_recorte(personPath):
     # inicia la deteccion y recorte 
    faceClassif = cv2.CascadeClassifier('C:/Users/yobis/Desktop/sistemabiors/SistemaBiometricoJessi/mysite/sistemabio/static/haarcascades/haarcascade_frontalface_default.xml')
    captureList = os.listdir(personPath)
    print('lista de imagenes', captureList)
    image_array = []
    count = 0
    for filename in captureList:
         imagepath = personPath+"/"+filename
         print(imagepath)
         image = cv2.imread(imagepath)
         if image is None: continue
         imageAux = cv2.imread(imagepath)
         gray = cv2.cvtColor(image,  cv2.COLOR_BGR2GRAY)
         faces = faceClassif.detectMultiScale(gray, 1.1, 5)
         for (x, y, w, h) in faces:
              cv2.rectangle(image, (x, y), (x + w, y + h), (128, 0, 255), 2)
              cv2.rectangle(image, (10, 5), (450, 25), (255, 255, 255), 2)
              rostro = imageAux[y:y + h, x:x + w]
              rostro = cv2.resize(rostro, (224, 224), interpolation=cv2.INTER_CUBIC)
              cv2.imwrite(personPath +'/'+ filename, rostro)
              print('leyendo imagenerecorte')
         imagepath = personPath+"/"+filename
         print(imagepath)
         image_file = open(imagepath, 'rb')
         image = image_file.read()
         print('finalizo deteccion y recorte ')
    return imagepath

def reconocimiento_facial(imagepath, personPath_2):
    print('inicio de reconocimiento prediccion ')
        
    images = []
    directories = []
    dircount = []
    prevRoot=''
    cant=0

    print("leyendo imagenes de ",personPath_2)

    for root, dirnames, filenames in os.walk(personPath_2):
        for filename in filenames:
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                cant=cant+1
                filepath = os.path.join(root, filename)
                image = plt.imread(filepath)
                images.append(image)
                b = "Leyendo..." + str(cant)
                print (b, end="\r")
                if prevRoot !=root:
                    print(root, cant)
                    prevRoot=root
                    directories.append(root)
                    dircount.append(cant)
                    cant=0
    dircount.append(cant)

    dircount = dircount[1:]
    dircount[0]=dircount[0]+1
    print('Directorios leidos:',len(directories))
    print("Imagenes en cada directorio", dircount)
    print('suma Total de imagenes en subdirs:',sum(dircount))


    #Se crean las etiquetas
    labels=[]
    indice=0
    for cantidad in dircount:
        for i in range(cantidad):
            labels.append(indice)
        indice=indice+1
    print("Cantidad etiquetas creadas: ",len(labels))

    deportes=[]
    indice=0
    for directorio in directories:
        name = directorio.split(os.sep)
        print(indice , name[len(name)-1])
        deportes.append(name[len(name)-1])
        indice=indice+1

    model = keras.models.load_model('facecop_mnist_jessi.h5py')
    # Cargar y preprocesar la nueva imagen que deseas predecir
    new_image_path = imagepath
    new_image = plt.imread(new_image_path)
    new_image = cv.resize(new_image, (224, 224))
    new_image = new_image.astype('float32') / 255.0
    new_image = np.expand_dims(new_image, axis=0)  # Agregar una dimensión extra para indicar el lote (batch)
    # Realizar la predicción en la nueva imagen
    predicted_probabilities = model.predict(new_image)
    predicted_class = np.argmax(predicted_probabilities)
    #Mostrar la clase predicha
    #print(f"La imagen se predice como la clase: {deportes[predicted_class]}")
    ##################################################3
    # Umbral para considerar una predicción válida
    umbral = 0.8  # Puedes ajustar este valor según tus necesidades
    predicted_probabilities = model.predict(new_image)
    predicted_class = np.argmax(predicted_probabilities)
    max_probability = np.max(predicted_probabilities)
    max_probability = np.round(max_probability,2)
    if max_probability >= umbral:
        # Si la probabilidad máxima es mayor o igual al umbral, la imagen se predice como una clase
        print(f"La imagen se predice como la clase verificacion: {deportes[predicted_class]}")
        print("probabilidad", max_probability)
        print('finalizaciom de reconocimiento prediccion Tryue ')

        return True
    else:
        # Si la probabilidad máxima es menor que el umbral, la imagen no coincide con ninguna clase
        print("La imagen no coincide con ninguna clase conocida.")
        print("probabilidad", max_probability)
        print('finalizaciom de reconocimiento prediccion False')
        return False

        # # Realizar la predicción en la nueva imagen
        # predicted_probabilities = model.predict(new_image)
        # predicted_class = np.argmax(predicted_probabilities)

        # Mostrar la clase predicha
    # print(f"La imagen se predice como la clase: ")

def accediste(request):
    title='accediste'
    return render (request,'sistemabio/accediste.html',{
         'mytitle':title
    })

def voz_usuario(request):
    title='voz_usuario'
    return render (request,'sistemabio/voz_usuario.html',{
         'mytitle':title
    })

def huella_usuario(request):
    title='huella_usuario'
    return render (request,'sistemabio/huella_usuario.html',{
         'mytitle':title
    })







# def voz3.0(request, usuario_id):
#      if request.method == "GET":
#          inquilino = get_object_or_404(Usuario,pk=usuario_id)
#          form= SesionFormVoz(instance=inquilino)
#          return render(request, 'sistemabio/vozjj.html', 
#                        {  'inquilino':inquilino,
#                           "form": form
#                         })
#      else :
#           form= SesionFormVoz(request.POST)
#           new_voz = form.save(commit=False)
#           if form.is_valid():
#             print("formulario", form.is_valid())
#             print('for', form['id_usuario'].value())
#             print('form ', form['id_tipo_sesion'].value())
#             dato = form['dato'].value()
#             inquilino = get_object_or_404(Usuario,pk=usuario_id)
#             print('id usuario: ', usuario_id)
#             print('usuario: ',inquilino.nombre, inquilino.ap_paterno, inquilino.ap_materno ) 
#             personName =  str(usuario_id) + inquilino.nombre + inquilino.ap_paterno + inquilino.ap_materno 
#             print("Nombre de personName es: ", personName)
#             dataPath = 'C:/Users/yobis/Desktop/sistemabio/mysite/sistemabio/static/inquilinos' + '/' + personName
#             personPath = dataPath + '/' + 'VOZ' + personName
#             print("Nombre de carpeta es: ", personPath)
#             if not os.path.exists(personPath):
#                  try:
#                       os.mkdir(personPath, mode=0o755)
#                       print('Carpeta creada: ',personPath)
#                  except OSError as e:
#                       if e.errno!=errno.EEXIST:
#                            raise
#             else :
#                  print('el directorio ya existe')
#             # Decodificar la cadena Base64
#             datos_decodificados = base64.b64decode(dato)
#             # Guardar los datos de audio en un archivo
#             voz_name = 'audio_user_'+ str(usuario_id) +'.wav'
#             print('voz_name: ',voz_name)
#             # Ahora, tienes los datos de audio en su formato original en el archivo "audio_original.wav"
#             with open(personPath+'/'+voz_name, 'wb') as audio_file:
#                  audio_file.write(datos_decodificados)
#             print('audio guardado')
#             captureList = os.listdir(personPath)
#             print('lista de voices', captureList)
#             voz_array = []
#             count = 0
#             for filename in captureList:
#                 vozpath = personPath+"/"+filename
#                 #vozpath = os.path.join(personPath, filename)
#                 print(vozpath)
#                 from pydub import AudioSegment
#                 # Cargar el archivo de audio en formato MP3 (o el formato que tengas)
#                 audio = AudioSegment.from_file(vozpath)
#                 # Ruta de salida para el archivo WAV
#                 output_wav_path = personPath + "/" + filename
#                 # Exportar el archivo a formato WAV
#                 audio.export(output_wav_path, format="wav")
#                 print(f"El archivo ha sido convertido a WAV y guardado en: {output_wav_path}")
#             #     # Leer el archivo WAV
#                 sample_rate, audio_data = wav.read(output_wav_path)
#                 # fs, Audiodata = wavfile.read('audio_user_21.wav')
#                 # signal, sr = torchaudio.load(vozpath)
#                 print('si leee')
#                 # Convertir a formato mono si es estéreo
#                 if len(audio_data.shape) > 1:
#                     audio_data = audio_data.mean(axis=1)
#                 # Frase específica que deseas comparar
#                 frase_especifica = "prueba del 31 de octubre del 23"
#                 # Identificar la voz humana usando SpeechRecognition
#                 r = sr.Recognizer()
#                 with sr.AudioFile(vozpath) as source:
#                     audio_text = r.record(source)
#                     try:
#                         recognized_text = r.recognize_google(audio_text, language='es-MX')
#                         print(f"Texto reconocido del archivo {filename}: {recognized_text}")
#                         # Comparar con la frase específica
#                         if recognized_text.lower() == frase_especifica.lower():
#                             print("Sí, es la frase correcta.")
#                             # Eliminar ruido utilizando pydub
#                             audio = AudioSegment.from_wav(vozpath)
#                             # Aplicar reducción de ruido (ajusta el valor según tus necesidades)
#                             audio = audio - 20
#                             # Guardar el audio limpio en un nuevo archivo
#                             audio_cleaned_path = os.path.join(personPath, 'cleaned_' + filename)
#                             audio.export(audio_cleaned_path, format="wav")
#                             print(f"Archivo de audio limpio guardado en: {audio_cleaned_path}")

#                             # Leer el archivo limpio en formato binario
#                             with open(audio_cleaned_path, "rb") as audio_file:
#                                 audio_bytes = audio_file.read()
#                             # Codificar en base64
#                             datos_codificados = base64.b64encode(audio_bytes)
#                             # print('Audio codificado en base64:', datos_codificados)
#                             new_voz.dato = datos_codificados
#                             print(new_voz.dato)
#                             new_voz.completado = True
#                             print(new_voz.completado)
#                             new_voz.save()
#                             print('Si se guardo el formulario')

#                             messages.success(request," El registro de voz ha sido un éxito.")
#                             return redirect('/sistemabio/inquilinos/')

#                         else:
#                             print("No es la frase correcta.")
#                     except sr.UnknownValueError:
#                         print(f"No se pudo reconocer la voz en el archivo {filename}.")
#                     except sr.RequestError as e:
#                         print(f"Error en la solicitud para el archivo {filename}: {e}")
#           else:
#                 messages.error(request, "Error no se creo el registro de voz.")
#                 return render(request, 'sistemabio/vozjj.html', 
#                               { 
#                                'inquilino': inquilino,
#                                "form":  form, 
#                                "error": "Error creando el registro de voz."})
#      # title='Voz'
#      # return render (request,'sistemabio/voz.html',{
#      #      'mytitle':title
#      # })
