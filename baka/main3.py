from transformers import pipeline

# Initialize the text generation pipeline with the GPT-2 model
generation = pipeline("text-generation", model="gpt2")

#baka coder using gpt 2 
prompt = "U area  good coding assiten to debug the cde and write a python hello world code for me"
code = generation(prompt, max_length=100, num_return_sequences=1)

print(code[0]["generated_text"])