<?php
/*
// Get the name of the uploaded file 
//$filename = $_FILES['file']['name'];
$filename = 'pic_'.date('YmdHis') . '.jpeg';
// Choose where to save the uploaded file 
$location = "faces/".$filename;

//Save the uploaded file to the local filesystem 
if ( move_uploaded_file($_FILES['file']['tmp_name'], $location) ) { 
  echo 'Success'; 
} else { 
  echo 'Failure'; 
}*/
/*
$url = '';
if( move_uploaded_file($_FILES['webcam']['tmp_name'],'upload/'.$filename) ){
   $url = 'http://' . $_SERVER['HTTP_HOST'] . dirname($_SERVER['REQUEST_URI']) . '/upload/' . $filename;
}*/


// new filename
$filename = 'pic_'.date('YmdHis') . '.jpeg';
$location = "faces/".$filename;
$url = '';
if( move_uploaded_file($_FILES['webcam']['tmp_name'],'upload/'.$filename) ){
   $url = 'http://' . $_SERVER['HTTP_HOST'] . dirname($_SERVER['REQUEST_URI']) . '/upload/' . $filename;
   //$url = "faces/".$filename;
}
/*
if ( move_uploaded_file($_FILES['webcam']['tmp_name'], $location) ) { 
  echo 'Success'; 
}*/
// Return image url
echo $url;

*/
?>
