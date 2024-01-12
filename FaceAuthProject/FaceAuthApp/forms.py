from django import forms
from .models import User

class signupForm(forms.ModelForm):
    class Meta:
        model=User
        fields=['username', 'email']

