from django.forms import ModelForm
from django import forms
from .models import Usuario, Sesion

class InquilinoForm(forms.ModelForm):
    # fecha_nac = forms.DateField(label ='Fecha Nacimiento',
    #                             widget =  forms.DateInput( 
    #                                 attrs={
    #                                     'class': 'form-control',
    #                                     'placeholder':  'MM-DD-YYYY',
    #                                         }
    #                                     ))
    
    # ap_paterno = forms.CharField(label = 'Apellido paterno', widget = forms.TextInput(attrs={
    #     'class': 'form-control',
    #     'placeholder':"Apellido Paterno ",
    # } ))
    # ap_materno = forms.CharField(label = 'Apellido materno', widget = forms.TextInput(attrs={
    #     'class': 'form-control',
    #     'placeholder':"Apellido PMaterno ",
    # } ))       
    # piso = forms.CharField(label = 'No. Piso:', widget = forms.TextInput(attrs={
    #     'class': 'form-control',
    #     'placeholder':"Número de Piso "
    # } ))
    # departamento = forms.CharField(label = 'No. Departamento:', widget = forms.TextInput(attrs={
    #     'class': 'form-control',
    #     'placeholder':"Número de Departamento "
    # } ))
    # telefono = forms.CharField(label = 'No. Teléfono:', widget = forms.TextInput(attrs={
    #     'class': 'form-control',
    #     'placeholder':"Telefono "
    # } ))
      
    class Meta:
        model = Usuario
        fields = ['nombre','ap_paterno','ap_materno', 'curp','piso','departamento','telefono', 'correo', 'fecha_nac', 'id_perfil', 'id_status']
        # self.fields['fecha_nac'].widget.attrs[]
     
        
    # def __init__(self, *args,**kwargs):
    #     super(InquilinoForm,self).__init__(*args,**kwargs)
    #     self.fields['id_perfil'].widget.attrs['class']= 'form-control'
    #     self.fields['nombre'].widget.attrs['class']= 'form-control'
    #     self.fields['nombre'].widget.attrs['placeholder']= "Nombre "
    #     self.fields['ap_paterno'].widget.attrs['class']= 'form-control'
        
    #     self.fields['curp'].widget.attrs['class']= 'form-control'
    #     self.fields['curp'].widget.attrs['placeholder']= "Curp "
        
    #     self.fields['correo'].widget.attrs['class']= 'form-control'
    #     self.fields['correo'].widget.attrs['placeholder']= "Correo "
        
    # #     
    # })
        # self.fields['fecha_nac'].widget.attrs['type']=  'date'
        # self.fields['fecha_nac'].widget.attrs['class']= 'form-control'
        # self.fields['fecha_nac'].widget.attrs['placeholder']=  'MM-DD-YYYY'
        
        # self.fields['id_status'].widget.attrs['class']= 'form-control'
                                              
   

class SesionForm(ModelForm):
    # opcion = forms.ChoiceField(choices=[('facial', 'Facial'), ('huella', 'Huella'), ('voz', 'Voz')])
    class Meta:
        model = Sesion
        fields = ['id_usuario', 'id_tipo_sesion', 'dato' ]
    def __init__(self, *args,**kwargs):
        super(SesionForm,self).__init__(*args,**kwargs)
        self.fields['id_tipo_sesion'].widget.attrs['class']='form-control'
        self.fields['id_usuario'].widget.attrs['class']= 'form-control'
        self.fields['dato'].widget.attrs['class']= 'form-control'
    



