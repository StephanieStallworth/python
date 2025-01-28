import streamlit as st
import plotly.express as px
from backend import get_data() # main.py and backend.py have to be in same directory for import to work

########## Widgets: Add title, text input, slider, selectbox, and subheader ##########
st.title("Weather Forecast for the Next Days")
place = st.text_input("Place:")

days = st.slider("Forecast Days:", min_value=1, max_value=5,
                 help="Select the number of forecated days")

option = st.selectbox("Select data to view",
                      ("Temperature", "Sky"))
st.subheader(f"{option} for the next {days} days in {place}")

##### Adding a Dynamic Graph #####
# Anytime user changes slider, code is executed from top to bottom
# Days gets "thrown" to the get_data function call
# Then "thrown" to the get_data function call as an argument
# Then "thrown" to the temperatures variable list comprehension
# Function returns the two values dates, temperatures
# That then gets rendered in the plot

# Move to its backend.py
# And call function from there instead
# def get_data(days):
#     dates = ["2022-25-10", "2022-26-10", "2022-27-10"]
#     temperatures = [10, 11, 15]
#     temperatures = [days * i for i in temperatures]
#     return dates, temperatures
# d, t = get_data(days)

# Fix so there is no error when user loads app for the first time
# When user loads app for the first time, 'place' will be an empty string
# Conditional block to execute code only if place exists as a string
if place:
    ######### Get the temperature/sky data #########
    # Get data based on number of days
    filtered_data = get_data(place, days
                    # , option
                    )
    ######### Create a temperature plot #########
    # Filter data further by temperature or sky conditions
    if option == "Temperature":
        temperatures = [dict["main"]["temp"] for dict in filtered_data]
        dates = [dict["dt_txt"] for dict in filtered_data]
        # Create a temperature plot
        figure = px.line(x=dates, # array object (list, Series, etc)
                         y=temperatures, # array same length as x
                         labels = {"x": "Date", "y": "Temperature (C)"}
                         )
        # Requires figure object as input
        # Get figure object from data visualization library (created above)
        # Then call function for that figure object
        st.plotly_chart(figure)

    # Render on web app
    if option == "Sky":
        # Create as many images as there are conditions
        images = {"Clear": "images/clear.png",
                  "Clouds": "images/clouds.png",
                  "Rain": "images/rain.png",
                  "Snow": "images/snow.png" }
        sky_conditions = [dict["weather"][0]["main"] for dict in filtered_data]
        # Translating the data
        image_paths = [images[condition] for condition in sky_conditions]
        print(sky_conditions)
        st.image(image_paths, width=115)