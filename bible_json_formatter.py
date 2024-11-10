import json
from book_mapping import book_number_to_name

# Load the JSON file
with open('kjv.json', 'r') as f:
    data = json.load(f)

# Extract verses with book names
verses = []
for row in data['resultset']['row']:
    fields = row['field']
    # Assuming fields: [id, book_number, chapter, verse, text]
    book_number = fields[1]
    chapter = fields[2]
    verse = fields[3]
    text = fields[4]
    book_name = book_number_to_name.get(book_number, f"Book {book_number}")  # Fallback in case of missing mapping
    verses.append(f"{book_name} {chapter}:{verse} - {text}")

# Save to a text file
with open('bible_text.txt', 'w') as f:
    for verse in verses:
        f.write(verse + "\n")

# Also save the verses list for semantic search
with open('verses_list.json', 'w') as f:
    json.dump(verses, f)
