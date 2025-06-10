from transformers import pipeline, AutoTokenizer, GPT2LMHeadModel

# Load trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("tiny-gpt")
tokenizer = AutoTokenizer.from_pretrained("tiny-gpt")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "The moon"
output = generator(prompt, max_new_tokens=50)[0]["generated_text"]

print("✨ Generated Text:\n", output)
