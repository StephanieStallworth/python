############### Character Classes: 12 Days of Christmas Example ##################
>>> lyrics = '12 drummers drumming, 11 pipers piping, 10 lords a-leaping, 9 ladies dancing, 8 maids a milking, 7 swans a swimming, 6 geese a-laying, 5 golden rings, 4 calling birds, 3 French hens, 2 turtle doves, and 1 partridge in a pear tree'

>>> # Identify pattern where we have some number followed by some words
>>> xmasRegex = re.compile(r'\d+\s\w+')

# Find all matches
>>> xmasRegex.findall(lyrics)
# ['12 drummers', '11 pipers', '10 lords', '9 ladies', '8 maids', '7 swans', '6 geese', '5 golden', '4 calling', '3 French', '2 turtle', '1 partridge']