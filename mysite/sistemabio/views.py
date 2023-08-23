from django.shortcuts import render

# Create your views here.

from django.http import HttpResponseRedirect,HttpResponse, Http404
from django.shortcuts import get_object_or_404, render, redirect, get_object_or_404
from django.urls import reverse
from django.views import generic
from django.utils import timezone
from django.template import loader
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
from .forms import InquilinoForm, SesionForm


#Administrador e Index
def index(request):
     title='Index'
    # return HttpResponse("Hello, world. You're at the polls index.")
     return render (request,"sistemabio/index.html",{
          'mytitle':title
     })
     
def signup(request):
        if request.method == 'GET':
            print('enviando formulario')
            title='Registrar'
            return render(request, "sistemabio/signup.html",{
                'mytitle':title,
                'form':UserCreationForm
            } ) 
        else:
             if request.POST["password1"] == request.POST["password2"]:
                  #register user
                  try:
                    user = User.objects.create_user(username=request.POST["username"],
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
             return render(request, "sistemabio/signup.html",{
                            'error': 'Passwords no coinciden',
                            'form':UserCreationForm
                        } )   
 

   #return render(request, "sistemabio/signin.html",
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
                return render(request, 'sistemabio/signin.html',{
                    'error': "usuario o password es incorrecto.",
                    'form': AuthenticationForm
                    })

              login(request, user)
              return redirect('/sistemabio/principal/')
            
   #return render(request, "sistemabio/signin.html",
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

def perfil_administrador(request):
     title='Perfil administrador'
     return render (request,"sistemabio/administradores/perfil.html",{
          'mytitle':title
     })
def administradores(request):
     title='Administradores'
     return render (request,"sistemabio/administradores/administradores.html",{
          'mytitle':title
     })

#Inquilinos
# def inquilinosinicial(request):
#      title='Inquilinos Inicial'
#      return render (request,"sistemabio/inquilinos/inquilinos_inicial_copy.html",{
#           'mytitle':title
#      })
def inquilinos(request):
     inquilinos = Usuario.objects.all()
     paginacion = Paginator(inquilinos,20)
     page_num = request.GET.get('page')
     page = paginacion.get_page(page_num)
     title='Inquilinos'
     sesiones = Sesion.objects.all()
     return render (request,"sistemabio/inquilinos/all-inquilinos.html",{
          'mytitle':title,
          'count': paginacion.count,
          'inquilinos':inquilinos,
          'sesiones': sesiones
     })
@login_required
def new_inquilino(request):
     if request.method == "GET":
        form=  InquilinoForm
        return render(request, 'sistemabio/inquilinos/new-inquilino.html', 
                      {"form":  InquilinoForm })   
     else:
        # print(request.POST)
        # return render(request, 'sistemabio/inquilinos/new-inquilino.html', {"form":  InquilinoForm})
        try:
            form = InquilinoForm(request.POST)
            new_inquilino = form.save(commit=False)
            new_inquilino.save()
            messages.success(request," El registro ha sido un éxito.")
            return redirect('/sistemabio/new_biometricos/')
        except ValueError:
            messages.error(request, "Error no se creo el inquilino.")
            return render(request, 'sistemabio/inquilinos/new-inquilino.html', 
                          {"form":  InquilinoForm , 
                           "error": "Error creando el inquilino."})
     
@login_required
def new_biometricos(request):
     if request.method == "GET":
        form= SesionForm
        return render(request, 'sistemabio/inquilinos/new-biometricos.html', 
                      {"form": SesionForm 
                       })   
     else:
        # print(request.POST)
        # return render(request, 'sistemabio/inquilinos/new-inquilino.html', {"form":  InquilinoForm})
        try:
            form = SesionForm(request.POST)
            new_biometricos = form.save(commit=False)
            new_biometricos.save()
            messages.success(request," El registro biométrico ha sido un éxito.")
            return redirect('/sistemabio/new_biometricos/')
        except ValueError:
            messages.error(request, "Error no se registro el biométrico.")
            return render(request, 'sistemabio/inquilinos/new-biometricos.html', 
                          {"form":  SesionForm,
                           "error": "Error registrando el biométrico."})
        
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
            Q(piso__icontains = busqueda) | 
            Q(nombre__icontains = busqueda) |
            Q(curp__icontains = busqueda) |
            Q(departamento__icontains = busqueda)
        ).distinct()  
     if nombre:
        inquilinos = Usuario.objects.filter(
            Q(nombre__icontains = nombre) 
        ).distinct()  
        print('nombre des ',nombre)
     if piso:
        inquilinos = Usuario.objects.filter(
            Q(piso__icontains = piso) 
        ).distinct() 
        print('npiso des ',piso) 
     if departamento:
        inquilinos = Usuario.objects.filter(
            Q(departamento__icontains = departamento)
        ).distinct() 
        print('ndepartamento des ',departamento)
        print('inquilino es ',inquilinos)
    # inquilino = get_object_or_404(Usuario,pk=inquilino_para)
    #  inquilinos = Usuario.objects.filter(nombre='jessica sanchez pruebaF5')
     title='search'
     return render (request,"sistemabio/inquilinos/s-inquilinos.html",{
          'mytitle':title,
          'inquilinos':inquilinos
     })
#     
def detail_inquilino(request, usuario_id):
    inquilino = get_object_or_404(Usuario,pk=usuario_id)
    print('usuario id ', usuario_id)
    title='detail'
    return render(request,"sistemabio/inquilinos/detail-inquilino.html",{
        'mytitle':title,
        'inquilino':inquilino
    })

def delete_inquilino(request, inquilino_id):
    inquilino = Usuario.objects.get( id_usuario=inquilino_id)
    if request.method == 'POST':
        inquilino.delete()
        return redirect('/sistemabio/inquilinos/')
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
               inquilino = get_object_or_404(Usuario,pk=usuario_id)
               form = InquilinoForm(request.POST,instance=inquilino)
               form.save()
               print('usuario id ', usuario_id)
               title='Edit Inquilino'
               messages.success(request,"Inquilino actualizado exitosamente")
               return redirect('/sistemabio/inquilinos/')
          # , "message": "Inquilino actualizado exitosamente"
          except ValueError:
            messages.error(request, "Error al actualizar el inquilino.")
            return render(request, 'sistemabio/inquilinos/edit-inquilino.html', 
                          {'inquilino': inquilino, "form":  InquilinoForm, "error": "Error actualizando el inquilino."})

# Datos biométricos
def facial(request):
     title='Facial'
     return render (request,'sistemabio/facial.html',{
          'mytitle':title
     })



    
