# Importing widget classes
from PyQT6.QtWidgets import QApplication,\
                # QVBoxLayout,
                QLabel, QWidget, QGridLayout,\
                 QLineEdit, QPushButton

import sys
from datetime import datetime

# Summary
# A Window is a represented by a class
# That class inherits from the QWidget class
# Inside the Child Class's __init__ method you construct that particular window
# Define how you want the window to look
# Add methods is you want to do other operations (calculations, processing of data)
# Then instantiate the Class and call the necessary methods at the end

##### Child Class of QWidget #####
# This is powerful than just creating an instance of QWidget
# Inherit from QWidget class so we can add more capabilities
class AgeCalculator(QWidget):

    ##### __init__ method to construct window #####
    def __init__(self):

        # __init__ of the child class overwrites __init__ of the parent class
        # Add this to call __init__ of the parent too
        super().__init__()

        # Create attributes/instance variables by adding "self" to the assignment line
        # Saying this its part of the Class Instance and can be access from the other methods we create below
        self.setWindowTitle("Age Calculator")

        # QGridLayout() gives you more flexibility on how to position the widgets, more powerful
        # QVBoxLayout() only lets you position widgets stacked on top of each other vertically
        # QHBoxLayout() allows you to position widgets horizontally
        grid = QGridLayout()

        ##### Create widgets #####
        name_label = QLabel("Name:")
        self.name_line_edit = QLineEdit()

        date_birth_label = QLabel("Date of Birth in MM/DD/YYYY:")
        self.date_birth_line_edit = QLineEdit()

        # Purpose of  the __init__  function is to construct the widgets, display the initial window to the user
        # Don't want to calculate age until the button is clicked
        calculate_button = QPushButton("Calculate Age")
        calculate_button.clicked.connect(self.calculate_age)

        self.output_label = QLabel("")

        ##### Add widgets to grid #####
        grid.addWidget(name_label, 0, 0)
        grid.addWidget(self.name_line_edit, 0, 1)
        grid.addWidget(date_birth_label, 1, 0)
        grid.addWidget(self.date_birth_line_edit, 1, 1) # have to change format everywhere it is referenced
        grid.addWidget(calculate_button, 2, 0, 1, 2)
        grid.addWidget(self.output_label, 3, 0, 1, 2)

        self.setLayout(grid)

    ##### Methods for Additional Operations #####
    def calculate_age(self):
        current_year = datetime.now().year
        date_of_birth = self.date_birth_line_edit.text() # Can be accessed from this method because we made it a Instance Variable above
        year_of_birth = datetime.strptime(date_of_birth, "%m/%d/%Y").date().year
        age = current_year - year_of_birth
        self.output_label.setText(f"{self.name_line_edit.text()} is {age} years old.")

##### Running loop of app #####
# Instantiate the class and call the necessary methods
# Routine calls that initialize the application
app = QApplication(sys.argv)
age_calculator = AgeCalculator()
age_calculator.show()
sys.exit(app.exec())