from datetime import datetime

from flask import Flask, render_template, request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message

# Create app instance using Flask class
app = Flask(__name__)

##### Creating a Database #####
# Create database instance
app.config["SECRET_KEY"] = "myapplication123" # guard against hackers, need this key to perform actions
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///data.db" # Specifies type of database & database file created by flask

##### Sending a Confirmation Email #####
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 465 # for Gmail
app.config["MAIL-USE_SSL"] = True
app.config["MAIL_USERNAME"] = "app8flask@gmail.com"
app.config["MAIL_PASSWORD"] = "soqlvetithqkagcx" # Password created from Google Account

db = SQLAlchemy(app)

# Connect it to the app
mail = Mail(app)

# Specify database model we want
class Form(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        first_name = db.Column(db.String(80))
        last_name = db.Column(db.String(80))
        email = db.Column(db.String(80))
        date = db.Column(db.Date)
        occupation = db.Column(db.String(80))

# Decorator to call function
@app.route("/", methods=["GET","POST"])
def index():
        # print(request.method)
        if request.method == "POST":
                first_name = request.form["first_name"]
                last_name = request.form["last_name"]
                email = request.form["email"]
                date = request.form["date"] # this is a string not Date type, wil throw a "StatementError" when we try to use as is
                date_obj = datetime.strptime(date, "%Y-%m-%d") # Create instance of datetime class from datetime library
                occupation = request.form["occupation"]

                ##### Storing the user data in the database #####
                # Performing "INSERT" SQL query in a high-level way
                # Form is the class we created above containing the database model
                form = Form(first_name=first_name, last_name=last_name,
                            email=email, date=date_obj, # using date object instead of original date variable
                            occupation=occupation)
                db.session.add(form)
                db.session.commit()

                ##### Send Email #####
                message_body = f"Thank you for your submission, {first_name}." \
                               f"Here are your data:\n{first_name}\n{last_name}\n{date}\n"\
                               f"Thank you!"

                message = Message(subject="New form submission",
                                  sender=app.config["MAIL_USERNAME"],
                                  recipients=[email],
                                  body=message_body
                                  )

                mail.send(message)

                ##### Showing Submissions Notification #####
                flash(f"{first_name}, your form was submitted successfully!", "success")

        return render_template("index.html")

if __name__ == "__main__":
        with app.app_context():
                db.create_all() # Actually create the database
                app.run(debug=True,
                        port=5001 # can put anything here
                        )
