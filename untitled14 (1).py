# -*- coding: utf-8 -*-
"""Untitled14.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12Kn3tWYxCS7REuDMBdoAKAKACZhklAO0
"""

import sqlite3

# Connect to SQLite (creates file if it doesn't exist)
conn = sqlite3.connect('sales_data.db')
cursor = conn.cursor()

# Create a table
cursor.execute('''
CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price REAL NOT NULL
)
''')

# Insert some sample data
sample_data = [
    ('Apples', 10, 0.50),
    ('Bananas', 5, 0.30),
    ('Apples', 20, 0.50),
    ('Oranges', 15, 0.60),
    ('Bananas', 7, 0.30),
    ('Oranges', 10, 0.60),
]

cursor.executemany("INSERT INTO sales (product, quantity, price) VALUES (?, ?, ?)", sample_data)
conn.commit()
conn.close()

import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('sales_data.db')

# Define the SQL query
query = '''
SELECT
    product,
    SUM(quantity) AS total_qty,
    SUM(quantity * price) AS revenue
FROM sales
GROUP BY product
'''

# Load query result into DataFrame
df = pd.read_sql_query(query, conn)

# Close the connection
conn.close()

# Print the DataFrame
print("Sales Summary:")
print(df)

import matplotlib.pyplot as plt

# Plot a bar chart for revenue by product
df.plot(kind='bar', x='product', y='revenue', legend=False)
plt.title('Revenue by Product')
plt.ylabel('Revenue ($)')
plt.xlabel('Product')
plt.tight_layout()

# Optional: Save the chart as an image
plt.savefig("sales_chart.png")
plt.show()