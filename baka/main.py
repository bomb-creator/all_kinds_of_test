from transformers import pipeline

# Using pre-trained transformers 
generator = pipeline("text-generation", model="distilgpt2")

# Generate a Toy Story 
prompt = "once a upon a time in a toy box there was a,"
story = generator(prompt, max_new_tokens=50, do_sample=True, pad_token_id=50256, truncation=True)

print(story[0]["generated_text"])