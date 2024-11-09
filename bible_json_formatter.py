import json

# Load the JSON file
with open('kjv.json', 'r') as f:
    data = json.load(f)

# Extract verses
verses = []
for row in data['resultset']['row']:
    fields = row['field']
    book = fields[1]
    chapter = fields[2]
    verse = fields[3]
    text = fields[4]
    verses.append(f"Book {book}, Chapter {chapter}, Verse {verse}: {text}")

# Save to a text file
with open('bible_text.txt', 'w') as f:
    for verse in verses:
        f.write(verse + "\n")
