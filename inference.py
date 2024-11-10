import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
import json

# Check for MPS availability
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load the fine-tuned model and tokenizer
model_path = "fine_tuned_llama"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# 2. Create a text generation pipeline
generator = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    device=0 if device.type == "mps" else -1
)

# 3. Load precomputed verse embeddings and verses
verse_embeddings = torch.load('verse_embeddings.pt')
with open('verses_list.json', 'r') as f:
    verses = json.load(f)

# 4. Load the embedding model for semantic search
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_search(query, top_k=5):
    """Perform semantic search to find relevant Bible verses."""
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, verse_embeddings, top_k=top_k)
    hits = hits[0]  # Get results for the first (and only) query
    return [verses[hit['corpus_id']] for hit in hits]

def generate_response_with_verses(user_input):
    """Generate AI response incorporating relevant Bible verses."""
    system_prompt = "You are an AI assistant created by Larson Carter from Carter Technologies, LLC. Provide detailed and theologically sound responses based on the Bible."
    prompt = f"{system_prompt}\nUser: {user_input}\nAI:"

    # Perform semantic search to retrieve relevant verses
    relevant_verses = semantic_search(user_input)
    if relevant_verses:
        verses_text = "\n".join(relevant_verses[:5])  # Limit to top 5 verses
        prompt += f"\nRelevant Bible Verses:\n{verses_text}\nAI:"

    # Generate the response
    response = generator(prompt, max_length=600, num_return_sequences=1)
    return response[0]['generated_text']

if __name__ == "__main__":
    print("Welcome to the Christian Pastor AI Assistant!")
    print("Type your question or topic below (type 'exit' to quit):\n")
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            print("Exiting the AI Assistant. Goodbye!")
            break
        response = generate_response_with_verses(user_input)
        print(f"\nAI Response:\n{response}\n{'-'*80}\n")
