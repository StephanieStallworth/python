from django.shortcuts import render
from django.views import generic
from .models import Item, MEAL_TYPE

# Create your views here.

##### Creating Class Based Views #####
# Variables have to have specific names that Django will look for
# Homepage representd by this view
class MenuList(generic.ListView):
    queryset = Item.objects.order_by("-date_created") # Item is class we created in models.py
    template_name = "index.html"

    ##### Context in Django #####
    # Have to name this method exactly like this
    # So we overwrite this pre-defined method by reusing name
    def get_context_data(self, **kwargs): # special parameter that allows you to handle named arguments that you have not defined in advance

        # Hardcoded values
        # context = {"meals":["Pizza","Pasta"],
        #            "ingredients": ["things"]
        #            }

        ##### Jinja For Loops #####
        # Dynamic values
        # Instead of starting with an empty dictionary
        # context = {}

        # Get dictionary from ListView Parent class
        context = super().get_context_data(**kwargs)

        # Then add more keys to that dictionary
        context["meals"] = MEAL_TYPE # tuple from models.py

        return context

# Individual item pages represented by this view
class MenuItemDetail(generic.DetailView):
    model = Item
    template_name = "menu_item_detail.html"



