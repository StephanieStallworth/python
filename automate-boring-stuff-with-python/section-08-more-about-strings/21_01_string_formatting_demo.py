# Normally when you want to combine strings into a single string can use the `+` operator to concatenate
>>> 'hello ' + 'world'
# 'hello world'

# But tricky if you have several strings that you need to concatenate
>>> name = 'Alice'
>>> place = 'Main Street'
>>> time = '6 pm'
>>> food = 'turnips'
>>> 'Hello ' + name + ', you are invited to a party at ' + place + ' at ' + time + '. Please bring ' + food + '.'
# 'Hello Alice, you are invited to a party at Main Street at 6 pm. Please bring turnips.'

# Instead you can use string formatting 
# Put a `%s` inside of a string to mark where we want to have other strings inserted
# Follow the string with a `%` 
# Then in parenthesis a comma-delimited list of variables that we want to have inserted at the `%s` placeholders (these are called "Conversion Specifiers")

>>> 'Hello %s, you are invited to a party at %s at %s. Please bring %s.'%(name, place, time, food)

# Evaluates to the same thing the huge string concatenation expression above evaluates to
# 'Hello Alice, you are invited to a party at Main Street at 6 pm. Please bring turnips.'