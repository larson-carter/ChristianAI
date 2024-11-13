import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import json
import os

# Set page configuration
st.set_page_config(page_title="Christian Pastor AI Assistant", layout="wide")

# Title and Description
st.title("Christian Pastor AI Assistant")
st.write("**Created by Larson Carter from Carter Technologies, LLC.**")


# Load model and tokenizer from Hugging Face
@st.cache_resource
def load_model():
    model_name = "larsoncarter/Christian-AI-LLAMA"  # Your model path
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")  # Fetch token if needed
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token).to(
        "cuda" if torch.cuda.is_available() else "cpu")

    # Create a text generation pipeline
    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    return generator


# Load embeddings and verses
@st.cache_resource
def load_embeddings():
    # Ensure 'verse_embeddings.pt' and 'verses_list.json' are in your GitHub repository
    verse_embeddings = torch.load('verse_embeddings.pt')
    with open('verses_list.json', 'r') as f:
        verses = json.load(f)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model, verse_embeddings, verses


generator = load_model()
embedding_model, verse_embeddings, verses = load_embeddings()


# Function for semantic search
def semantic_search(query, top_k=5):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, verse_embeddings, top_k=top_k)
    hits = hits[0]  # Get the first (and only) query's results
    return [verses[hit['corpus_id']] for hit in hits]


# Function to generate response
def generate_response(user_input):
    system_prompt = (
        "You are an AI assistant created by Larson Carter from Carter Technologies, LLC. "
        "Your task is to provide detailed, theologically accurate, and well-structured responses based on the Bible. "
        "When referencing Bible verses, use the proper book names followed by chapter and verse numbers (e.g., Genesis 1:1). "
        "Ensure that your responses are clear, concise, and free from repetition."
    )
    prompt = f"{system_prompt}\nUser: {user_input}\nAI:"

    # Semantic search for relevant verses
    relevant_verses = semantic_search(user_input)
    if relevant_verses:
        verses_text = "\n".join(relevant_verses[:5])  # Limit to top 5 verses
        prompt += f"\nRelevant Bible Verses:\n{verses_text}\n"

    # Generate response with enhanced parameters
    response = generator(
        prompt,
        max_length=800,  # Increased length for completeness
        num_return_sequences=1,
        num_beams=5,  # Beam search for better coherence
        no_repeat_ngram_size=3,  # Prevents repetition
        early_stopping=True,
        temperature=0.7,  # Controls randomness
        top_p=0.9  # Controls diversity
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

# Feedback Section
st.markdown("---")
st.header("Feedback")

feedback = st.text_area("We value your feedback! Please let us know your thoughts or any issues you encountered:")

if st.button("Submit Feedback"):
    if feedback.strip() != "":
        # Option 1: Save feedback to a file (Not recommended for scalability)
        with open("feedback.txt", "a") as f:
            f.write(feedback + "\n")
        st.success("Thank you for your feedback!")
    else:
        st.warning("Please enter your feedback before submitting.")
