from django.forms import ModelForm
from .models import Usuario, Sesion

class InquilinoForm(ModelForm):
    class Meta:
        model = Usuario
        fields = ['nombre','ap_paterno','ap_materno', 'curp','piso','departamento','telefono', 'correo', 'fecha_nac', 'id_perfil', 'id_status']
    def __init__(self, *args,**kwargs):
        super(InquilinoForm,self).__init__(*args,**kwargs)
        self.fields['id_perfil'].widget.attrs['class']= 'form-control'
        self.fields['nombre'].widget.attrs['class']= 'form-control'
        self.fields['nombre'].widget.attrs['placeholder']= "Nombre "
    #     
    # })
        self.fields['fecha_nac'].widget.attrs['type']=  'date'
        self.fields['fecha_nac'].widget.attrs['class']= 'form-control'
        self.fields['fecha_nac'].widget.attrs['placeholder']=  'MM-DD-YYYY'
        
        self.fields['id_status'].widget.attrs['class']= 'form-control'
                                              
   

class SesionForm(ModelForm):
    class Meta:
        model = Sesion
        fields = ['id_usuario', 'id_tipo_sesion', 'dato' ]
    # def __init__(self, *args,**kwargs):
    #     super(SesionForm).__init__(*args,**kwargs)
    #     self.fields['id_tipo_sesion'].widget.attrs.update({
    #         'class': 'form-control'
    #     })
    #     self.fields['id_usuario'].widget.attrs.update({
    #         'class': 'form-control'
    #     })
    #     self.fields['dato'].widget.attrs.update({
    #         'class': 'form-control'
    #     })



