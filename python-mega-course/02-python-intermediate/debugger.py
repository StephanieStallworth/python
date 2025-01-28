########## Example 1: No Functions ###############
text = "I love sun"
word = "love"

words = text.split( )
count = text.count(word)

frequency = count/len(words)
print(frequency)

########## Example 2: Functions ##########
message = "Hello"
import glob

def get_frequency(text, word):
    words = text.split(" ")
    glob.glob("*")
    count = text.count(word)
    frequency = count / len(words) * 100
    return frequency

frequency = get_frequency("I love sun", "love")
if frequency > 5:
    print("High frequency")
else:
    print("Low frequency")

print(message)