from django.contrib import admin
from .models import Item

# Register your models here.
class MenuItemAdmin(admin.ModelAdmin):
    list_display = ("meal", "status") # database fields
    list_filter = ("status", ) # extra comma to make it a tuple
    search_fields = ("meal","description")

# Item class from models.py is coupled with class above then register
admin.site.register(Item, MenuItemAdmin)
