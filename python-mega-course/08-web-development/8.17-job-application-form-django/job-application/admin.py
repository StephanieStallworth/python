from django.contrib import admin
from .models import Form

##### Customizing The Admin Interface #####
# Modify default model admin
class FormAdmin(admin.ModelAdmin):
    list_display = ("first_name", "last_name", "email")  # fields from models.py
    search_fields = ("first_name", "last_name", "email")
    list_filter = ("date","occupation")
    ordering = ("-first_name", ) # Need comma to make it a tuple, add "-"  to make reverse order
    readonly_fields = ("occupation", ) #

##### Creating An Admin Interface  #####
# Register your models here.
# Expects Class to register with Admin interface
admin.site.register(Form, FormAdmin)
