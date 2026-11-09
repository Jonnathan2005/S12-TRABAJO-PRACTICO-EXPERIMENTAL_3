from django import forms

from .models import Persona

FORMATOS_IMAGEN = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
TAMANO_MAXIMO = 8 * 1024 * 1024  # 8 MB


class PersonaForm(forms.ModelForm):
    class Meta:
        model = Persona
        fields = ['nombres', 'apellidos', 'cedula', 'correo', 'curso', 'activo']
        widgets = {
            'nombres': forms.TextInput(attrs={'placeholder': 'Ej. María José'}),
            'apellidos': forms.TextInput(attrs={'placeholder': 'Ej. Vera Castro'}),
            'cedula': forms.TextInput(attrs={'placeholder': '10 dígitos', 'inputmode': 'numeric', 'maxlength': 10}),
            'correo': forms.EmailInput(attrs={'placeholder': 'correo@unemi.edu.ec'}),
            'curso': forms.TextInput(attrs={'placeholder': 'Ej. Software 7mo A2'}),
        }

    def clean_nombres(self):
        return self.cleaned_data['nombres'].strip().title()

    def clean_apellidos(self):
        return self.cleaned_data['apellidos'].strip().title()


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleImageField(forms.FileField):
    widget = MultipleFileInput

    def clean(self, data, initial=None):
        archivos = data if isinstance(data, (list, tuple)) else [data]
        archivos = [a for a in archivos if a]
        if not archivos:
            raise forms.ValidationError('Seleccione al menos una imagen.')
        for archivo in archivos:
            validar_imagen(archivo)
        return archivos


def validar_imagen(archivo):
    if not archivo.name.lower().endswith(FORMATOS_IMAGEN):
        raise forms.ValidationError(f'"{archivo.name}" no es una imagen válida (jpg, png, bmp, webp).')
    if archivo.size > TAMANO_MAXIMO:
        raise forms.ValidationError(f'"{archivo.name}" supera los 8 MB.')


class SubirMuestrasForm(forms.Form):
    fotos = MultipleImageField(label='Fotos del rostro')



class FiltroAsistenciaForm(forms.Form):
    desde = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date'}))
    hasta = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date'}))
    persona = forms.ModelChoiceField(queryset=Persona.objects.all(), required=False, empty_label='Todas')
    estado = forms.ChoiceField(required=False, choices=[('', 'Todos'), ('puntual', 'Puntual'), ('atraso', 'Atraso')])
