
import cv2

#import cv2
import base64
import sys
# ************** Abrir camara **************
""" 
cap= cv2.VideoCapture(0)
while cap.isOpened():
    ret, im= cap.read()

    if ret == False:
        break
    cv2.imshow('imagen', im)

    if cv2.waitKey(1) == 27:
        break

"""
# ************** Lectura de imagenes **************

"""im =cv2.imread('logo.png')
cv2.imshow('imagen',im)
cv2.waitKey(3600) # Se espera 360milisegundos"""

# ************** Tomar foto **************

cap = cv2.VideoCapture(0) # abrir la cámara

# establecer dimensiones
cap.set(cv2.CAP_PROP_FRAME_WIDTH,2560) # ancho
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1440) # alto

# Tomar una imagen
ret, frame = cap.read()

# Guardamos la imagen en un archivo
cv2.imwrite('rostro15.jpg',frame)

#{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}
# Codificar la imagen en base64
with open("rostro15.jpg", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

# Devolver la imagen codificada en base64 a la consola
sys.stdout.write(encoded_image)
sys.stdout.flush()

#Liberamos la cámara
cap.release()
cv2.imshow('Imagen Capturada',frame)
cv2.waitKey(3600) # Se espera 360milisegundos