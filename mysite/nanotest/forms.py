from django import forms
from .models import *
from django.db import models
from django.db.models import fields
from django.forms import NumberInput
from django.forms import widgets, ModelForm

class FeatureForm(forms.ModelForm):
    class Meta:
        model = Features
        fields = "__all__"
        widgets = {
            'val1':  NumberInput(attrs={'class': 'form-control'}),
            'val2':  NumberInput(attrs={'class': 'form-control'}),
            'val3':  NumberInput(attrs={'class': 'form-control'}),
            'val4':  NumberInput(attrs={'class': 'form-control'}),
            'val5':  NumberInput(attrs={'class': 'form-control'}),
            'val6':  NumberInput(attrs={'class': 'form-control'}),
            'val7':  NumberInput(attrs={'class': 'form-control'}),
            'val8':  NumberInput(attrs={'class': 'form-control'}),
            'val9':  NumberInput(attrs={'class': 'form-control'}),
            'val10':  NumberInput(attrs={'class': 'form-control'}),
        }