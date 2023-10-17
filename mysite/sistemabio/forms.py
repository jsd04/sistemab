from django.forms import ModelForm
from django import forms
from .models import Usuario, Sesion

class InquilinoForm(forms.ModelForm): 
    class Meta:
        model = Usuario
        fields = ['nombre','ap_paterno','ap_materno', 'curp','piso','departamento','telefono', 'correo', 'fecha_nac', 'id_perfil', 'id_status']
        widtgets={
             'fecha_nac':forms.TextInput(attrs={'class': 'form-control'})
            }
        # acepta fechas aaaa-mm-dd y dd/mm/aaaa pero no aaaa/mm/dd ni dd-mm-aaaa 
        # self.fields['fecha_nac'].widget.attrs[]
    def __init__(self, *args,**kwargs):
        super(InquilinoForm,self).__init__(*args,**kwargs)
        self.fields['nombre'].widget.attrs['class']= 'form-control'
        self.fields['ap_paterno'].widget.attrs['class']= 'form-control'
        self.fields['ap_materno'].widget.attrs['class']= 'form-control'
        self.fields['curp'].widget.attrs['class']= 'form-control'
        self.fields['piso'].widget.attrs['class']= 'form-control'
        self.fields['departamento'].widget.attrs['class']= 'form-control'
        self.fields['telefono'].widget.attrs['class']= 'form-control'
        self.fields['correo'].widget.attrs['class']= 'form-control'
        self.fields['fecha_nac'].widget.attrs['class']= 'form-control'
        self.fields['id_perfil'].widget.attrs['class']= 'form-control'
        self.fields['id_status'].widget.attrs['class']= 'form-control'
        self.fields['ap_paterno'].label= 'Apellido Paterno'
        self.fields['ap_materno'].label='Apellido Materno'
        self.fields['piso'].label='No. Piso'
        self.fields['departamento'].label= 'No. Departamento'
        self.fields['telefono'].label= 'No. Teléfono'
        self.fields['fecha_nac'].label= 'Fecha de Nacimiento'
        self.fields['id_perfil'].label= 'Perfil de Usuario'
        self.fields['id_status'].label= 'Status'
        
                                              
   
#solo usuario
class SesionForm(ModelForm):
    class Meta:
        model = Sesion
        fields = ['id_usuario']
    def __init__(self, *args,**kwargs):
         super(SesionForm,self).__init__(*args,**kwargs)
        # self.fields['id_tipo_sesion'].widget.attrs['class']='form-control'
         self.fields['id_usuario'].widget.attrs['class']= 'form-control'
#solo dato
class SesionForm2(ModelForm):
    class Meta:
        model = Sesion
        fields = ['dato' ]
    def __init__(self, *args,**kwargs):
        super(SesionForm2,self).__init__(*args,**kwargs)
        
        self.fields['dato'].widget.attrs['class']= 'form-control'
#todo el modelo
class SesionForm3(ModelForm):
    class Meta:
        model = Sesion
        fields = ['id_usuario', 'dato','id_tipo_sesion','completado']
        labels = {'completado':'Completo',}
    def __init__(self, *args,**kwargs):
         super(SesionForm3,self).__init__(*args,**kwargs)
         #Widgets son los que se van a pintan en forma de etiquetas html
         self.fields['id_usuario'].widget.attrs['class']= 'form-control' 
         self.fields['id_usuario'].label= 'Usuario'
         self.fields['dato'].widget.attrs['class']='form-control' 
         self.fields['id_tipo_sesion'].widget.value='1'
         self.fields['id_tipo_sesion'].widget.attrs['class']='form-control' 
         self.fields['id_tipo_sesion'].label= 'Tipo Session'
         #self.fields['completado'].label= 'Hecho'


