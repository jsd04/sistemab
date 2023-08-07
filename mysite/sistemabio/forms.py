from django.forms import ModelForm
from .models import Usuario

class InquilinoForm(ModelForm):
    class Meta:
        model = Usuario
        fields = ['nombre','ap_paterno','ap_materno', 'curp','piso','departamento','telefono', 'correo', 'fecha_nac', 'id_perfil', 'id_status']



