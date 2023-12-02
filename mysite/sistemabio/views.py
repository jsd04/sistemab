from django.shortcuts import get_object_or_404, render, redirect, get_object_or_404
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.urls import reverse
import base64
import os
import errno
#creatiionform
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from django.contrib.auth import login, logout, authenticate
from django.db import IntegrityError
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.contrib import messages
from django.core.paginator import Paginator
from .models import Usuario, Sesion
from .forms import InquilinoForm, SesionForm, SesionFormUsuario, SesionForm3, SesionFormVoz

#Administrador e Index
def index(request):
     title='Index'
     return render (request,"sistemabio/index2.html",{
          'mytitle':title
     })  
def home_admin(request):
     title='Index Admin'
     return render (request,"sistemabio/index.html",{
          'mytitle':title
     })  
def home_usuario(request):
     title='Index Usuario'
     return render (request,"sistemabio/index3u.html",{
          'mytitle':title
     })

def signup(request):
        if request.method == 'GET':
            print('enviando formulario')
            title='Registrar'
            return render(request,"sistemabio/signup.html",
                         {
                              'mytitle':title,
                              'form':UserCreationForm
                         } ) 
        else:
             if request.POST["password1"] == request.POST["password2"]:
                  #register user
                  try:
                    user = User.objects.create_user(username=request.POST["username"],
                                                    email=request.POST["email"],
                                                    first_name=request.POST["first_name"], 
                                                    last_name=request.POST["last_name"],
                                                   password=request.POST["password1"])
                    user.save()
                    login(request, user)
                    return redirect('/sistemabio/principal/')
                   # return HttpResponse('User creado satisfactoriamente')        
                  except IntegrityError:
                       return render(request, "sistemabio/signup.html",{
                            'error': 'El user ya existe',
                            'form':UserCreationForm
                        } ) 
             messages.error(request,  'Passwords no coinciden')    
             return render(request, "sistemabio/signup.html",{
                            'error': 'Passwords no coinciden',
                            'form':UserCreationForm
                        } )   
 

def signin(request):
         if request.method == "GET":
            title='Iniciar sesion'
            return render(request, "sistemabio/signin.html",{
                'mytitle':title,
                'form':AuthenticationForm
            } )        
         else:
              user = authenticate( request, username=request.POST['username'], password=request.POST['password'])
              if user is None:
                messages.error(request, "Usuario o password es incorrecto")
                return render(request, 'sistemabio/signin.html',{
                    'error': "usuario o password es incorrecto.",
                    'form': AuthenticationForm
                    })

              login(request, user)
              return redirect('/sistemabio/principal/')
            
@login_required
def principal(request):
     title='Principal Administrador'
     return render (request,"sistemabio/principal_2.html",{
          'mytitle':title
     })
@login_required
def signout(request):
     logout(request)
     messages.warning(request,"Estás desconectado ahora. Inicia sesión")
     return redirect('/sistemabio/signin/')
def about(request):
     title='About'
     return render (request,"sistemabio/about.html",{
          'mytitle':title
     })
@login_required
def perfil_administrador(request):
     title='Perfil Administrador'
     return render (request,"sistemabio/perfil.html",{
          'mytitle':title
     })
# Inquiinos
def inquilinos(request):
     inquilinos = Usuario.objects.all().order_by('id_usuario')
     paginator = Paginator(inquilinos, 5)  # Divide los datos en páginas con 5 elementos por página
     page_number = request.GET.get('page')
     try:
         page_obj = paginator.page(page_number)
     except PageNotAnInteger:
         # Si el número de página no es un entero, muestra la primera página
         page_obj = paginator.page(1)
     except EmptyPage:
         # Si la página está fuera de rango (por ejemplo, página 9999), muestra la última página
         page_obj = paginator.page(paginator.num_pages)
     title='Inquilinos'
     sesiones = Sesion.objects.all()
     return render (request,"sistemabio/inquilinos/all-inquilinos.html",{
          'mytitle':title,
          'page_obj': page_obj,
          'sesiones': sesiones
     })

def new_inquilino(request):
     if request.method == "GET":
        form=  InquilinoForm
        return render(request, 'sistemabio/inquilinos/new-inquilino.html', 
                      {"form":  InquilinoForm })   
     else:
        # print(request.POST)
        # return render(request, 'sistemabio/inquilinos/new-inquilino.html', {"form":  InquilinoForm})
        # commit=False es para procesar los datos antes de guardar, se usa cuando no tienes todos tus 
        # campos llenos de ese form   new_inquilino = form.save()  new_inquilino.save()
        try:
            form = InquilinoForm(request.POST)
            if form.is_valid():
               
               form.save()
            print("Formulario es ", form.is_valid())
            messages.success(request," El registro ha sido un éxito.")
            inquilino = get_object_or_404(Usuario, nombre=form['nombre'].value())
            print('inquilino: ', inquilino.id_usuario)
            personName = str(inquilino.id_usuario) + inquilino.nombre + inquilino.ap_paterno + inquilino.ap_materno
            print("Nombre de personName es: ", personName)
            dataPath = 'C:/Users/yobis/Desktop/sistemabio/mysite/sistemabio/static/inquilinos'
            personPath = dataPath + '/' + personName
            print("Nombre de carpeta es: ", personPath)
            try:
                 os.mkdir(personPath, mode=0o755)
                 print('Carpeta creada: ',personPath)
            except OSError as e:
                 if e.errno!=errno.EEXIST:
                      raise
            return redirect('/sistemabio/new_biometricos/')
        except ValueError:
            messages.error(request, "Error no se creo el inquilino.")
            return render(request, 'sistemabio/inquilinos/new-inquilino.html', 
                          {"form":  InquilinoForm , 
                           "error": "Error creando el inquilino."})
            
def search_inquilino(request):
     #buscar por tipo de usuario
     busqueda = request.POST.get("buscar")
     print('nusqueda es ',busqueda)
     nombre = request.POST.get("nombre")
     print('nombre es ',nombre)
     piso = request.POST.get("piso")
     print('npiso es ',piso)
     departamento = request.POST.get("departamento")
     print('ndepartamento es ',departamento)
     inquilinos = Usuario.objects.all()
     if busqueda:
        inquilinos = Usuario.objects.filter(
            Q(piso = busqueda) | 
            Q(nombre = busqueda) |
            Q(curp = busqueda) |
            Q(departamento = busqueda)
        ).distinct()  
     if nombre:
        inquilinos = Usuario.objects.filter(
            Q(nombre = nombre) 
        ).distinct()  
        print('nombre des ',nombre)
     if piso:
        inquilinos = Usuario.objects.filter(
            Q(piso = piso) 
        ).distinct() 
        print('npiso des ',piso) 
     if departamento:
        inquilinos = inquilinos.filter(
            Q(departamento = departamento)
        ).distinct() 
        print('ndepartamento des ',departamento)
        print('inquilino es ',inquilinos)
     title='search'
     return render (request,"sistemabio/inquilinos/s-inquilinos.html",{
          'mytitle':title,
          'inquilinos':inquilinos
     })
#     
def detail_inquilino(request, usuario_id):
    title='detail'
    inquilino = get_object_or_404(Usuario,id_usuario=usuario_id)
    sesiones =  Sesion.objects.all().filter(id_usuario_id=usuario_id).values() 
#    sesiones =  Sesion.objects.filter(id_usuario=1).values('id_sesion', 'id_usuario_id', 'id_tipo_sesion', 'completado', 'dato', 'fecha_creacion', 'fecha_actualizacion')
    print('usuario id ', usuario_id)
    print(sesiones)
    print('................')
    python6423 = None  # Valor predeterminado
    for sesion in sesiones:
        if sesion['id_tipo_sesion'] == '1':
     #    print('dato: ',sesion['id_sesion']) para acceder a los campos es con '' dentro de corchetes
          print('=======================')   
          print('dato: ',sesion['dato'])
          pytho = base64.b64encode(sesion['dato'])
          print('-------------------------------')
          print(pytho)
          python642 = pytho.decode('utf-8')
          print('+++++++++++++++++++++++++++++++++')
          print(python642)
          datos_div = python642.split()
          i=0
          for dato_div in datos_div:
               variable = datos_div[i]
               print('************************************************************')
               print('cadena dividida: ',variable)
               python6423 = 'data:image/jpg;base64,' + str(variable)
               print('***********************')
               print('uuuuuuuuuu',python6423)
    if python6423:
        return render(request, "sistemabio/inquilinos/detail-inquilino.html", {
            'mytitle': title,
            'inquilino': inquilino,
            'sesiones': sesiones,
            'python6423': python6423,
        })
    else:
        return render(request,"sistemabio/inquilinos/detail-inquilino.html",{
          'mytitle':title,
          'inquilino':inquilino,
          'sesiones':sesiones,
        })



def delete_inquilino(request, inquilino_id):
    inquilino = Usuario.objects.get( id_usuario=inquilino_id)
    if request.method == 'POST':
        inquilino.delete()
        return redirect('/sistemabio/inquilinos/')
    return render(request,"sistemabio/inquilinos/delete-inquilino.html",{
        'inquilino':inquilino
    })
@login_required
def edit_inquilino(request, usuario_id):
     if request.method == "GET":
          inquilino = get_object_or_404(Usuario,pk=usuario_id)
          form = InquilinoForm(instance=inquilino)
          return render(request,"sistemabio/inquilinos/edit-inquilino.html",{
               'inquilino':inquilino,
               'form':form})
     else:
          try:
               inquilino = Usuario.objects.get(pk=usuario_id)
               form = InquilinoForm(request.POST,instance=inquilino)
               # commit=False es para procesar los datos antes de guardar, se usa cuando no tienes todos tus 
               # campos llenos de ese form   
                    # update_inquilino = form.save(commit=False) 
                    # update_inquilino.save()
               # if form.is_valid():
               form.save()
               print('Edición de usuario: ', usuario_id)
               messages.success(request,"Inquilino actualizado exitosamente")
               print("Formulario es ", form.is_valid())
               return redirect('/sistemabio/inquilinos/')
          except ValueError:
            messages.error(request, "Error al actualizar el inquilino.")
            return render(request, 'sistemabio/inquilinos/edit-inquilino.html', 
                          {'inquilino': inquilino, "form":  InquilinoForm, "error": "Error actualizando el inquilino."})

# LA SIGUIENTE FUNCION USO DE ESTA VIEWS, TAMPOCO LAS ESTOY USANDO
def ingresar_usuario(request):
    title='ingresar_usuario'
    return render (request,'sistemabio/ingresar.html',{
         'mytitle':title
    })



# LA SIGUIENTE FUNCION NO LA ESTOY USANDO, USO LA MISMA DE NEW_INQUILINO
def registrar_usuario(request):
    title='registrar_usuario'
    return render (request,'sistemabio/registrar.html',{
         'mytitle':title
    })


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


    
