from sentence_transformers import SentenceTransformer, util
import json
import torch

# Load the Bible data with book names
with open('verses_list.json', 'r') as f:
    verses = json.load(f)

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and efficient

# Create embeddings
print("Creating embeddings for Bible verses...")
verse_embeddings = embedding_model.encode(verses, convert_to_tensor=True, show_progress_bar=True)

# Save embeddings and verses
torch.save(verse_embeddings, 'verse_embeddings.pt')
with open('verses_list.json', 'w') as f:
    json.dump(verses, f)

print("Embeddings and verses saved successfully.")
