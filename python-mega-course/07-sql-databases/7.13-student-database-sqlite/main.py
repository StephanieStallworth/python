# Importing widget classes
from PyQT6.QtWidgets import QApplication,  QLabel, QWidget, QGridLayout,\
                 QLineEdit, QPushButton, QMainWindow, QTableWidget, QTableWidgetItem, QDialog, \
                QVBoxLayout, QComboBox, QToolBar
from PyQT6.QTGui import QAction
import sys
import sqlite3


########## Create main window with a menu bar ##########
# QMainWindow allows us to add a menu bar, tool bar, and status bar
# Can have division among the different sections of the app
# So use QMainWindow when you have a larger app instead of QWdidget
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__() # instatiate __init__ parent method
        self.setWindowTitle("Student Management System")

        file_menu_item = self.menuBar().addMenu("&File")
        help_menu_item = self.menuBar().addMenu("&Help")


        # Add sub-item for each menu item (dropdown options)
        add_student_action = QAction("Add Student", self) # connects QAction to the actual class
        add_student_action.triggered.connect(self.insert) # insert() method created below
        file_menu_item.addAction(add_student_action)

        about_action = QAction("About", self)
        help_menu_item.addAction(about_action)

        # If help item didn't display (MacOS)
        about_action.setMenuRole(QAction.MenuRole.NoRole)

        ########## Creating a Table Structure ##########
        self.table = QTableWidget() # Need to make this an attirbute instead of instance
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(("Id", "Name", "Course", "Mobile"))
        self.table.verticalHeader().setVisible(False) # Hide index column
        self.setCentralWidget(self.table) # Specify table as the central widget, not using layout as its used when you're using QWidget and have multiple layouts

        # Could call method here
        # self.load_data()

        ########## Add the Toolbar ##########
        # Create toolbar and add toolbar elements
        toolbar = QToolBar()
        toolbar.setMovable(True)
        self.addToolBar(toolbar)
        toolbar.addAction(add_student_action)




        ########## Populate Table with Data ##########
        # Since these are functions of the sqlite3 library, make sense to call them in a separate method here
        # Can call this method anywhere, in the class definition (above) or running loop part of the app (below)
        def load_data(self):
            connection = sqlite3.connect("database.db") # create database file using DB Browser for SQLite program
            result = connection.execute("SELECT * FROM students") # extract data from database
            self.table.setRowCount(0) # Resets the table and loads the data as fresh to avoid duplicates
            # print(list(result))  # Returns list of tuples that represents rows
            for row_number, row_data in enumerate(result):
                self.table.insertRow(row_number) # inserting empty row in particular index of table
                for column_number, data in enumerate(row_data):
                    self.table.setItem(row_number, column_number, QTableWidgetItem(str(data)))
            connection.close()

        ########## Inserting New Records ##########
        # Create InsertDialog Class object (defined below) that is displayed on the screen as a pop up window
        def insert(self):
            dialog = InsertDialog()
            dialog.exec() #

class InsertDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Insert Student Data")

        # Setting fixed width and height a good practice for dialog windows
        self.setFixedWidth(300)
        self.setFixedHeight(300)

        # Using layout instead of grid since app prototype is structured vertically
        layout = QVBoxLayout()

        # Create the widgets
        # aConvert to instance variable so add_student() method below can access it using "self"
        # Add student name widget
        self.student_name = QLineEdit()
        self.student_name.setPlaceholderText("Name")
        layout.addWidget(self.student_name)

        # Add combo box of courses
        self. course_name = QComboBox()
        courses = ["Biology", "Math", "Astronomy", "Physics"]
        self.course_name.addItems(courses)
        layout.addWidget(self.course_name)

        # Add mobile widget
        self.mobile = QLineEdit()
        self.mobile.setPlaceholderText("Mobile")
        layout.addWidget(self.mobile)

        # Add a submit button
        button = QPushButton("Register")
        button.clicked.connect(self.add_student)

        self.setLayout(layout)

    # Create database file with program such as DB Browser for SQLite
    # Can set configuation to automatically add unique "id" column to new records that are inserted
    def add_student(self):
        name = self.student_name.text()
        course = self.course_name.itemText(self.course_name.currentIndex()) # Gives you the choice user made in the combo box
        mobile = self.mobile.text()
        connection = sqlite3.connect("database.db")
        cursor = connection.cursor() # need cursor object this time since we are inserting data not just viewing it
        cursor.execute("INSERT INTO students (name, course, mobile) VALUES (?, ?, ?)",
                       (name, course, mobile) # variables have to be in a tuple as this is one argument
                       )
        connection.commit() # apply SQL query to the database
        cursor.close() # close cursor
        connection.close() # close connection

        # Load the data by referring to instance of MainWindow class
        # And calling the load_data() method from it
        main_window.load_data()

##### Running loop of app #####
# Instantiate the class and call the necessary methods
# Routine calls that initialize the application
# "age_calculator" refactored to more meaningful name "main_window"
app = QApplication(sys.argv)
main_window = MainWindow()
main_window.show()
main_window.load_data() # Could also call it here
sys.exit(app.exec())

