from django import forms

# Copied from models.py
# Duplicate code but manual is better if we need to make changes to one
class ApplicationForm(forms.Form):
    first_name = forms.CharField(max_length=80) # update to forms module
    last_name = forms.CharField(max_length=80)
    email = forms.EmailField()
    date = forms.DateField()
    occupation = forms.CharField(max_length=80)


