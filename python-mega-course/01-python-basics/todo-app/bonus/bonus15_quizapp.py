import json

with open("bonus/questions.json","r") as file:
    content = file.read()

# read() gives us the content as a single string, not useful at all
# The loads ("load string") function from the json module will convert string to a list that we can extract elements from (not possible with just the string)
# When you read files will just give you a string
# If you put it in a json file, can convert that string to the Python data objects they are supposed to be

data = json.loads(content)

# print(type(content))
# print(type(data))

# Nested for-loop under mother for-loop

for question in data:
    print(question["question_text"])
    for index, alternative in enumerate(question["alternatives"]):
        print(index + 1, "-", alternative)
    user_choice = int(input("Enter your answer: "))
    question["user_choice"] = user_choice

# Show answer user made, parse data to make it more readable
# To split an f-string, place line continuation character "\" where you want to split (no spaces after!!!!0 and hit ENTER
# Make sure both parts of the string are in double quotes and begin with: f"

score = 0
for index, question in enumerate(data):
    if question["user_choice"] == question["correct_answer"]:
        score = score + 1
        result = "Correct Answer"
    else:
        result = "Wrong Answer"

    message = f" {index + 1}  {result}- Your answer: {question['user_choice']}, "\
             f"Correct answer: {question['correct_answer']}"  # have to use single quotes here because outer quotes are double quotes
    print(message)

# print(data)
print(score, "/", len(data))





