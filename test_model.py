from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import json

# Load device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
model_path = "fine_tuned_llama"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# Create generation pipeline
generator = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    device=0 if device.type == "mps" else -1
)

# Load embeddings and verses
verse_embeddings = torch.load('verse_embeddings.pt')
with open('verses_list.json', 'r') as f:
    verses = json.load(f)

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def semantic_search(query, top_k=5):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, verse_embeddings, top_k=top_k)
    hits = hits[0]
    return [verses[hit['corpus_id']] for hit in hits]


def generate_response_with_verses(user_input):
    system_prompt = "You are an AI assistant created by Larson Carter from Carter Technologies, LLC. Provide detailed and theologically sound responses based on the Bible."
    prompt = f"{system_prompt}\nUser: {user_input}\nAI:"

    # Semantic search
    relevant_verses = semantic_search(user_input)
    if relevant_verses:
        verses_text = "\n".join(relevant_verses[:5])
        prompt += f"\nRelevant Bible Verses:\n{verses_text}\nAI:"

    # Generate response
    response = generator(prompt, max_length=600, num_return_sequences=1)
    return response[0]['generated_text']


# Example test cases
test_queries = [
    "How can I teach about forgiveness in my sermon?",
    "What does the Bible say about faith?",
    "Can you help me write a sermon on love?",
    "Explain the concept of salvation.",
    "How to discuss the importance of prayer?"
]

for query in test_queries:
    print(f"User: {query}\n")
    response = generate_response_with_verses(query)
    print(f"AI: {response}\n{'-' * 80}\n")
