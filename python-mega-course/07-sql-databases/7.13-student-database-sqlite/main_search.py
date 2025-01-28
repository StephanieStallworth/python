from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QGridLayout, \
     QLineEdit, QPushButton, QMainWindow, QTableWidget, QTableWidgetItem, QDialog, \
     QVBoxLayout, QComboBox, QComboBox, QToolBar,QStatusBar, QMessageBox
from PyQt6.QtGui import QAction, QIcon
import sys
import sqlite3

######### Refactoring Fix: Duplicate Code ########
# Create class that takes care of the database connection so only have to mention once
# Make it an attribute of this class
# Then all classes should use instance of this class
class DatabaseConnection:
    def __init__(self, database_file = "database.db"): # only have to mention file once here
        self.database_file = database_file # make it an attribute so other methods in this class can access

    def connect(self):
        connection = sqlite3.connect(self.database_file)
        return connection

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Student Management System")

        self.setMinimumSize(800, 600)

        file_menu_item = self.menuBar().addMenu("&File")
        help_menu_item = self.menuBar().addMenu("&Help")
        edit_menu_item = self.menuBar().addMenu("&Edit")

        ##### Add Icon #####
        add_student_action = QAction(QIcon("icons/add.png"),"Add Student", self)
        add_student_action.triggered.connect(self.insert)
        file_menu_item.addAction(add_student_action)

        about_action = QAction("About", self)
        help_menu_item.addAction(about_action)
        about_action.setMenuRole(QAction.MenuRole.NoRole)

        # Connect object to method
        about_action.triggered.connect(self.about)

        ##### Search Icon #####
        search_action = QAction(QIcon("icons/search.png"), "Search", self)
        edit_menu_item.addAction(search_action)
        search_action.triggered.connect(self.search)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(("Id", "Name", "Course", "Mobile"))
        self.table.verticalHeader().setVisible(False)
        self.setCentralWidget(self.table)

        ########## Add the Toolbar ##########
        # Create toolbar and add toolbar elements
        toolbar = QToolBar()
        toolbar.setMovable(True)
        self.addToolBar(toolbar)

        # Creating icons above
        toolbar.addAction(add_student_action)
        toolbar.addAction(search_action)

        ########## Status Bar ##########
        # Create status bar and add status bar elements
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        # Show text in status bar
        # Can make this dynamic from SQL query
        # hello = QLabel("Hello There!")
        # statusbar.addWidget(hello)
        # hello = QLabel("Hello World!")
        # statusbar.addWidget(hello)

        # Detect a cell click
        # To dynamically add widgets to status bar, connect it here
        # Then define method below
        self.table.cellClicked.connect(self.cell_clicked)

    # Define method that is caleld above
    # Write conditionals and generate widgets when something happens
    def cell_clicked(self):

        # Edit button
        edit_button = QPushButton("Edit Record")
        edit_button.clicked.connect(self.edit)

        # Delete button
        delete_button = QPushButton("Delete Record")
        delete_button.clicked.connect(self.delete)

        # Clear current buttons if any are found
        children = self.findChildren(QPushButton)
        if children:
            for child in children:
                self.statusbar.removeWidget(child)

        # Add buttons to status bar
        self.statusbar.addWidget(edit_button)
        self.statusbar.addWidget(delete_button)

    def load_data(self):
        # connection = sqlite3.connect("database.db")
        connection = DatabaseConnection().connect()
        result = connection.execute("SELECT * FROM students")
        self.table.setRowCount(0)
        for row_number, row_data in enumerate(result):
            self.table.insertRow(row_number)
            for column_number, data in enumerate(row_data):
                self.table.setItem(row_number, column_number, QTableWidgetItem(str(data)))
        connection.close()

    def insert(self):
        dialog = InsertDialog()
        dialog.exec()

    def search(self):
        dialog = SearchDialog()
        dialog.exec()

    def edit(self):
        dialog = EditDialog()
        dialog.exec()

    def delete(self):
        dialog = DeleteDialog()
        dialog.exec()

    def about(self):
        dialog = AboutDialog()
        dialog.exec()

########## Create an About Dialog ##########
class AboutDialog(QMessageBox): # inheriting from QMessageBox class as this will be a simple window
    def __init__(self):
        super().__init__()
        self.setWindowTitle("About")
        content = """
        This app was created during the course "The Python Megate Course. 
        Feel free to modify and resuse this app.
        """
        self.setText(content)


########## Create an Edit Dialog ##########
class EditDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Update Student Data")
        self.setFixedWidth(300)
        self.setFixedHeight(300)

        layout = QVBoxLayout()

        # Get student name from selected row
        index = main_window.table.currentRow()
        student_name = main_window.table.item(index, 1).text()

        # Add student name widget
        self.student_name = QLineEdit("John Smith")
        self.student_name.setPlaceholderText("Name")
        layout.addWidget(self.student_name)

        # Add combo box of courses
        course_name = main_window.table.item(index, 2).text()
        self.course_name = QComboBox()
        courses = ["Biology", "Math", "Astronomy", "Physics"]
        self.course_name.addItems(courses)
        self.course_name.setCurrentText(course_name)
        layout.addWidget(self.course_name)

        # Add mobile widget
        mobile = main_window.table.item(index, 3).text()
        self.mobile = QLineEdit(mobile)
        self.mobile.setPlaceholderText("Mobile")
        layout.addWidget(self.mobile)

        # Add a submit button
        button = QPushButton("Update")
        button.clicked.connect(self.update_student)
        layout.addWidget(button)

        self.setLayout(layout)

    def update_student(self):
        # connection = sqlite3.connect("database.db")
        # Repeating this line isntead of creating a variable that the methods refer to because we want the connection established and closed with each of the classes
        connection = DatabaseConnection().connect()
        cursor = connection.cursor() # cursor object because we want to modify the database
        cursor.execute("UPDATE students SET name = ?, course = ?, mobile = ? WHERE id = ?",
                       (self.student_name.text(),
                        self.course_name.itemText(self.course_name.currentIndex()),
                        self.mobile.text(),
                        self.student_id)
                       ) # use cursor to execute SQL query

        # Commit changes as UPDATE and INSERT are write operations
        # Don't have to do this with SELECT query because it's not an write operation
        connection.commit()
        cursor.close()
        connection.close()

        # Refresh the table
        main_window.load_data()

########## Create a Delete Dialog ##########
class DeleteDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Update Student Data")
        # self.setFixedWidth(300)
        # self.setFixedHeight(300)

        layout = QGridLayout()
        confirmation = QLabel("Are you sure you want to delete?")
        yes = QPushButton("Yes")
        no =  QPushButton("No")

        layout.addWidget(confirmation, 0, 0, 1, 2)
        layout.addWidget(yes, 1, 0)
        layout.addWidget(no, 1, 1)
        self.setLayout(layout)

        # Method to execute when button is clicked
        yes.clicked.connect(self.delete_student)

    def delete_student(self):
        # Get selected row index and student id
        index = main_window.table.currentRow() # could have also put this in the __init__ method
        student_id = main_window.table.item(index, 0).text() # don't need "self" here unless line above was put in the __init_ method

        # connection = sqlite3.connect("database.db")
        connection = DatabaseConnection().connect()
        cursor = connection.cursor()
        cursor.execute("DELETE from students WHERE id = ?",
                       (student_id, ) # need comma so it is read as a tuple
                       )
        connection.commit() # because this is a write operation
        cursor.close()
        connection.close()
        main_window.load_data()

        self.close()

        # Confirmation messages
        # Creating a class instance here, not inheriting from the class
        # Which is fine, can do either way, but inheriting tends to make the code more organized
        confirmation_widget = QMessageBox()
        confirmation_widget.setWindowTitle("Success")
        confirmation_widget.setText("The record was deleted successfully!")
        confirmation_widget.exec()


class InsertDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Insert Student Data")
        self.setFixedWidth(300)
        self.setFixedHeight(300)

        layout = QVBoxLayout()

        # Get student name from selected row
        index = main_window.table.currentRow()
        student_name = main_window.table.item(index, 1).text()

        # Get id from selected row
        self.student_id = main_window.table.item(index, 0).text()

        # Add student name widget
        self.student_name = QLineEdit()
        self.student_name.setPlaceholderText("Name")
        layout.addWidget(self.student_name)

        # Add combo box of courses
        self.course_name = QComboBox()
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
        layout.addWidget(button)

        self.setLayout(layout)

    def add_student(self):
        name = self.student_name.text()
        course = self.course_name.itemText(self.course_name.currentIndex())
        mobile = self.mobile.text()
        # connection = sqlite3.connect("database.db")
        connection = DatabaseConnection().connect()
        cursor = connection.cursor()
        cursor.execute("INSERT INTO students (name, course, mobile) VALUES (?, ?, ?)",
                       (name, course, mobile))
        connection.commit()
        cursor.close()
        connection.close()
        main_window.load_data()


class SearchDialog(QDialog):
    def __init__(self):
        super().__init__()
        # Set window title and size
        self.setWindowTitle("Search Student")
        self.setFixedWidth(300)
        self.setFixedHeight(300)

        # Create layout and input widget
        layout = QVBoxLayout()
        self.student_name = QLineEdit()
        self.student_name.setPlaceholderText("Name")
        layout.addWidget(self.student_name)

        # Create button
        button = QPushButton("Search")
        button.clicked.connect(self.search)
        layout.addWidget(button)

        self.setLayout(layout)

    def search(self):
        name = self.student_name.text()
        # connection = sqlite3.connect("database.db")
        connection = DatabaseConnection().connect()
        cursor = connection.cursor()
        result = cursor.execute("SELECT * FROM students WHERE name = ?", (name,))
        row = list(result)[0]
        print(row)
        items = main_window.table.findItems("John Smith", Qt.MatchFlag.MatchFixedString)
        for item in items:
            print(item)
            main_window.table.item(item.row(), 1).setSelected(True)

        cursor.close()
        connection.close()


app = QApplication(sys.argv)
main_window = MainWindow()
main_window.show()
main_window.load_data()
sys.exit(app.exec())