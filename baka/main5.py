from transformers import pipeline

genreratotr = pipeline ("text-generation", model="distilgpt2")

# Using pre-trained transformers
prompt = "once a upon a time in a toy box there was a,"
story = genreratotr(prompt, max_new_tokens=50, do_sample=True, pad_token_id=50256, truncation=True)

# Generate a Toy Story
print(story[0]["generated_text"])
