import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Check for MPS availability
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load the fine-tuned model and tokenizer
model_path = "fine_tuned_llama"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# Create a generation pipeline
generator = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    device=0 if device.type == "mps" else -1
)

def generate_response(user_input):
    system_prompt = "You are an AI assistant created by Larson Carter from Carter Technologies, LLC. Provide detailed and theologically sound responses based on the Bible."
    prompt = f"{system_prompt}\nUser: {user_input}\nAI:"
    response = generator(prompt, max_length=600, num_return_sequences=1)
    return response[0]['generated_text']

# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("Enter your question (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        response = generate_response(user_input)
        print("\nAI Response:\n")
        print(response)
        print("\n" + "-"*80 + "\n")
