import sqlite3

# Establish a connection and a cursor object
connection = sqlite3.connect("data.db")
cursor = connection.cursor()

# Query all data based on a condition
# Modifies the cursor object to contain what the query returns
cursor.execute("SELECT * FROM events WHERE date='2088.10.15'")
rows = cursor.fetchall()
print(rows)

# Query certain columns based on a condition
cursor.execute("SELECT band, date FROM events WHERE date='2088.10.15'")
rows = cursor.fetchall()
print(rows)

# Insert new rows
# Need to prepare list of tuples
new_rows = [('Cats', 'Cat City', '2088.10.17'),
            ('Hens', 'Hens City', '2088.10.17')]
cursor.executemany("INSERT INTO events VALUES(?,?,?)", new_rows)
connection.commit()

# Query all data
cursor.execute("SELECT * FROM events")
rows = cursor.fetchall()
print(rows)