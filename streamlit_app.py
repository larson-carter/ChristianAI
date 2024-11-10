import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import json

# Set page configuration
st.set_page_config(page_title="Christian Pastor AI Assistant", layout="wide")

# Title and Description
st.title("Christian Pastor AI Assistant")
st.write("**Created by Larson Carter from Carter Technologies, LLC.**")

# Load model and tokenizer
@st.cache_resource
def load_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("fine_tuned_llama")
    model = AutoModelForCausalLM.from_pretrained("fine_tuned_llama").to(device)
    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=0 if device.type == "mps" else -1
    )
    return generator, device

# Load embeddings and verses
@st.cache_resource
def load_embeddings():
    verse_embeddings = torch.load('verse_embeddings.pt')
    with open('verses_list.json', 'r') as f:
        verses = json.load(f)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model, verse_embeddings, verses

generator, device = load_model()
embedding_model, verse_embeddings, verses = load_embeddings()

# Function for semantic search
def semantic_search(query, top_k=5):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, verse_embeddings, top_k=top_k)
    hits = hits[0]  # Get the first (and only) query's results
    return [verses[hit['corpus_id']] for hit in hits]

# Function to generate response
def generate_response(user_input):
    system_prompt = "You are an AI assistant created by Larson Carter from Carter Technologies, LLC. Your task is to provide detailed, theologically accurate, and well-structured responses based on the Bible. When referencing Bible verses, use the proper book names followed by chapter and verse numbers (e.g., Genesis 1:1). Ensure that your responses are clear, concise, and free from repetition."
    prompt = f"{system_prompt}\n\n\nUser: {user_input}\n\n\nAI:"

    # Semantic search for relevant verses
    relevant_verses = semantic_search(user_input)
    if relevant_verses:
        verses_text = "\n".join(relevant_verses[:5])
        prompt += f"\nRelevant Bible Verses:\n\n\n{verses_text}\n\n\nAI:"

    # Generate the response
    response = generator(
        prompt,
        max_length=800,                # Increased length
        num_return_sequences=1,
        num_beams=5,                   # Beam search
        no_repeat_ngram_size=3,        # Prevent repetition
        early_stopping=True
    )

    return response[0]['generated_text']

# User Input
user_input = st.text_input("Enter your question or topic:", "")

# Generate Button
if st.button("Generate Response"):
    if user_input.strip() != "":
        with st.spinner("Generating response..."):
            ai_response = generate_response(user_input)
        st.success("AI Response:")
        st.write(ai_response)
    else:
        st.warning("Please enter a valid question or topic.")
