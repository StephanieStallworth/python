from django.urls import path
from . import views # import views.py as a module inside this file

urlpatterns = [
    # Home Page
    path('', # Leave as empty string for home page
         views.index, # function it points to
         name = 'index' # function name as a string
         ),

    # About page
    path('about/', views.about, name='about')
]

