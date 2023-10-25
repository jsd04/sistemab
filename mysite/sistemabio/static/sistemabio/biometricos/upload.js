const fs = require('fs');
const path = require('path');

// Nuevo nombre de archivo
const filename = 'pic_' + new Date().toISOString().replace(/[-:.]/g, '') + '.jpeg';

// Ruta de la carpeta de destino
const uploadPath = path.join(__dirname, 'upload');

// Lee los datos de la imagen desde $_FILES['webcam']['tmp_name']
const imageData = fs.readFileSync('ruta_de_la_imagen.jpeg'); // Reemplaza 'ruta_de_la_imagen.jpeg' con la ruta real de la imagen

// Guarda la imagen en la carpeta de destino
fs.writeFileSync(path.join(uploadPath, filename), imageData);

// Genera la URL de la imagen
const url = 'http://' + req.headers.host + '/upload/' + filename; // Asegúrate de ajustar '/upload/' a la ruta correcta si es diferente

// Devuelve la URL de la imagen
console.log(url);
