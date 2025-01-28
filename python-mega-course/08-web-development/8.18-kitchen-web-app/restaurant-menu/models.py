from django.db import models
from django.contrib.auth.models import User # Class that represents

MEAL_TYPE = (
    ("starters","Starters"),
    ("salads","Salads"),
    ("main_dishes", "Main Dishes"),
    ("desserts", "Desserts")
)

STATUS = (
    (0, "Unavailable"),
    (1, "Available")
)
# Create your models here.
class Item(models.Model):
    meal = models.CharField(max_length=1000, unique=True)
    description = models.CharField(max_length=2000)
    price = models.DecimalField(max_digits=10,decimal_places=2)
    meal_type = models.CharField(max_length=200, choices=MEAL_TYPE)

    # Create many-to-one relationship between User table and Item table (one cook can make many meals)
    author = models.ForeignKey(User,
                               on_delete = models.PROTECT # Don't delete meals if user is deleted
                               # on_delete=models.CASCADE # Delete associated meals when user is deleted
                               # on_delete = model.SET_NULL # Set meal's author field NULL when user is deleted
                               )

    status = models.IntegerField(choices=STATUS, default=1)
    date_created = models.DateTimeField(auto_now_add=True) # Records datetime stamp when item is added
    date_updated = models.DateTimeField(auto_now=True) # Records when item is updated

    def __str__(self):
        return self.meal