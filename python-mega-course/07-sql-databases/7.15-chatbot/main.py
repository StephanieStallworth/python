from PYQT6.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton
import sys
from backend import Chatbot
import threading

########## Creating the ChatBot GUI ##########
# Front end with 3 widgets
class ChatbotWindow(QMainWindow):
    def __init__(self):
        super().__init__() # Instantiate parent class

        # import from backend.py, better to instantiate here instead of in a method
        # Like starting the conversation once then keep asking questions instead of a new call each time
        # Make it an attribute by adding "self"
        self.chatbot = Chatbot()

        self.setMinimumSize(700, 500) # pixels

        # Add chat area widget
        self.chat_area = QTextEdit(self) # use "self" to add to main window
        self.chat_area.setGeometry(10, 10, 480, 320)
        self.chat_area.setReadOnly(True)

        # Add the input field widget
        self.input_field = QLineEdit(self)
        self.input_field.setGeometry(10, 340, 480, 40)
        self.input_field.returnPressed.connect(self.send_message) # Enter/Send button

        # Add the button
        self.button = QPushButton("Send", self)
        self.button.setGeometry(500, 340, 100, 40)
        self.button.clicked.connect(self.send_message) # new method created below

        # Display
        self.show()

        ########## Connect the ChatBot to the GUI ##########
        # Method that is called above
        def send_message(self):
            user_input = self.input_field.text().strip()
            self.chat_area.append(f"<p style='color:#333333'>Me: {user_input}</p>") # throws it as a argument when get_response() is called below
            self.input_field.clear()

            ##### Threading #####
            # Use threading for better user experience, since it takes time to get a response
            thread = threading.Thread(target=self.get_bot_response, # target method to be called
                                      args=(user_input,) # tuple with 1 item, make sure there is a comma here
                                      )

            # This part moved to get_bot_response() method below so it can be called from a separate thread

            ## import from backend.py, better to instantiate in the __init__ method then make it an attribute
            ## chatbot = Chatbot()

            # throws to a parameter in get_response() which assigns it to a local variable "prompt" then throws result back here
            # response = self.chatbot.get_response(user_input)

            # Display in chat area
            # self.chat_area.append(f"<p style='color:#333333; background-color: #E9E9E9'>Bot:{response}</p>")

        def get_bot_response(self, user_input):
            response = self.chatbot.get_response(user_input)
            self.chat_area.append(f"<p style='color:#333333; background-color: #E9E9E9'>Bot:{response}</p>")


########## Create the GPT ChatBot ##########
# Move to backend.py
# class Chatbot:
#     pass

########## Execute App ##########
# Instantiate the actual application
app = QApplication(sys.argv)

# Instantiate the window itself
main_window = ChatbotWindow()

# Exit
sys.exit(app.exec())