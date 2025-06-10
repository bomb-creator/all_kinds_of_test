from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

prompt = "Write a short poem about the moon on a wedding night."

poem = generator(
    prompt,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.9,   # more creativity
    top_k=50,          # limit to top 50 word choices
    top_p=0.95         # nucleus sampling
)

print(poem[0]["generated_text"])
