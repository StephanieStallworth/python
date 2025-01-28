# main.py
from flask import Flask, render_template
import pandas as pd

# Good practice to use "__name__" for website name instead of "Website"
# So we can control when Flask website is run (at the bottom)
app = Flask(__name__)

stations = pd.read_csv("data_small/stations.txt", skiprows=17)
stations = stations[["STAID","STANAME                                 "]]

# Create instance of a  website object that represents a website
# Technically this is a "Flask object"
# The @ symbol means line is a "decorater" that connects method to this function
# Flask is configured to look for HMTL files in 'templates' folder of root directory
# When user visits the URL the function is called
# Need to change the local URL to view specific page that is rendered: http://127.0.0.1:5000/home

@app.route("/") # home page, not a subpage
def home():
    return render_template("home.html", # will look for this in templates folder
                           data=stations.to_html() # Show station table on the homepage
                           )

@app.route("/api/v1/<station>/<date>") # http://127.0.0.1:5000/api/v1/10/1988-10-25
def about(station,date):
    filename = "data_small/TG_STAID" + str(station).zfill(6) + ".txt"
    df = pd.read_csv(filename, skiprows=20, parse_dates=["    DATE"])
    temperature = df.loc[df['    DATE'] ==date]['   TG'].squeeze()/10
    return {"station": station,
            "date": date,
            "temperature": temperature}

# Add URL endpoint for all data for a station
@app.route("/api/v1/<station>") # http://127.0.0.1:5000/api/v1/10 (no slash at the end)
def all_data(station):
    filename = "data_small/TG_STAID" + str(station).zfill(6) + ".txt"
    df = pd.read_csv(filename, skiprows=20, parse_dates=["    DATE"])
    result = df.to_dict(orient="records") # more readable format
    return result

# Add URL endpoint for one year for a station
@app.route("/api/v1/yearly/<station>/<year>")
def yearly(station,year):
    filename = "data_small/TG_STAID" + str(station).zfill(6) + ".txt"
    df = pd.read_csv(filename, skiprows=20
                     # , parse_dates=["    DATE"] removing so Pandas will read as date
                     )
    df["    DATE"] = df["    DATE"].astype(str)
    result = df[df["    DATE"].str.startswith(str(year))].to_dict(orient="records")
    return result

# When a script is imported into another script and indirectly executed that way, its "__name__" variable will be the name of the script
# If script is executed directly, it's "__name__" variable will be "__main__"
# Can use this to our advantage to control when the Flask app is run
# Only run Flask app when this main.py script is executed directly
# If we're importing this script into another script want to just use the functions, not run the app
if __name__ == "__main__":
    app.run(debug=True)