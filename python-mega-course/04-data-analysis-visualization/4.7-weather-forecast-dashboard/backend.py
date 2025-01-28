import requests

API_KEY = "141710af2113bab9f55ef73e1bcd33d5"

# Return temperature or sky conditions given the forecast days
def get_data(place, forecast_days=None
             # , kind=None # not doing filtering based on this option so removing this
             ):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={place}&appid={API_KEY}"
    response = requests.get(url)
    data = response.json()
    filtered_data = data["list"]
    nr_values = 8 * forecast_days
    filtered_data = filtered_data[:nr_values]
    # Do the filtering in main.py now
    # if kind == "Temperature":
        # filtered_data = [dict["main"]["temp"] for dict in filtered_data]
    # if kind == "Sky":
        # filtered_data = [dict["weather"][0]["main"] for dict in filtered_data]
    return filtered_data

# Conditional block if code is executing directl,y
# User provided in main.py but pass in values here to test out
if __name__=="__main__":
    print(get_data(place="Tokyo", forecast_days=3
                   # , kind="Termpature" # not doing filtering based on this option so removing this
                   )
          )