import fs from "fs"


//'use strict';
//const fs = require('fs');

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const snap = document.getElementById("snap");
const errorMsgElement = document.querySelector('span#errorMsg');

const constraints = {
  audio: false,
  video: {
    width: 420, 
    height: 340,
    image_format: 'jpeg',
    jpeg_quality: 90
  }
};

// Access webcam
async function init() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    handleSuccess(stream);
  } catch (e) {
    errorMsgElement.innerHTML = `navigator.getUserMedia error:${e.toString()}`;
  }
}

// Success
function handleSuccess(stream) {
  window.stream = stream;
  video.srcObject = stream;
}

// Load init
init();
var numPhotos = 10; // Número total de fotos que se van a capturar
var photoCounter = 0; // Contador de fotos capturadas
var i=0;
// Draw image
var context = canvas.getContext('2d');
snap.addEventListener("click", function() { 
  

  const path="C:/Users/yobis/Desktop/Proyectos/pt/SistemaBiometricoPT/Interfaz/Administrador/src/public/fotos/";
/*
  try {
    if (!fs.existsSync(path)) {
      fs.mkdirSync(path);
    }
  } catch (err) {
   
    console.error(err);
    fs.mkdirSync(path+i);
    i++;
  }*/
 
const folderName = '/Users/joe/test'
/*
try {
  if (!fs.existsSync(folderName)) {
    fs.mkdirSync(folderName)
  }
} catch (err) {
  console.error(err)
}*/

  for (var i = 0; i <numPhotos; i++) {
        context.drawImage(video, 0, 0, 420, 340);
        // Generar un nombre de archivo único para la imagen
        var fileName = 'imagen_' + photoCounter + '.png';

        // Guardar la imagen localmente con el nombre de archivo generado
     //   var saveUrl = path + fileName; // Ruta y nombre de archivo para guardar la imagen
       

        // Incrementar el contador de fotos capturadas
        photoCounter++;
        console.log("Foto tomada ", fileName)
        uploadFile();

}
async function uploadFile() {
  let formData = new FormData(); 
  formData.append("file", fileupload.files[0]);
  await fetch('/upload.php', {
    method: "POST", 
    body: formData
  }); 
  alert('The file has been uploaded successfully.');
  }
console.log("Total de Fotos tomadas ", photoCounter)
});
