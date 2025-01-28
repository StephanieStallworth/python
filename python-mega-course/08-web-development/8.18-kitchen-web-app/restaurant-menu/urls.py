from django.urls import path
from . import views

# Use exact variable name
urlpatterns = [
    path('' # Should be an empty string for the home page so there is nothing after the URL (i.e., mysite.com)
         , views.MenuList.as_view() # When user visits homepage, calls method on MenuList class from views.py that sends index.html to the user's browser
         , name='home' # URL name, used in other parts of the Django app
         ),
    ###### Adding Dynamic Links #####
    path('item/<int:pk>/', # Make link more dynamic, expects integer from Primary Key of database
         views.MenuItemDetail.as_view(), # Should match associated function in views.py,
         name='menu_item' # Name jinja will look for to get value
         )

]
