from transformers import pipeline, AutoTokenizer, GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("tiny-gpt")
tokenizer = AutoTokenizer.from_pretrained("tiny-gpt")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

print("🧠 TinyGPT Chatbot — type 'exit' to quit\n")

while True:
    prompt = input("You: ")
    if prompt.strip().lower() == "exit":
        break
    output = generator(prompt, max_new_tokens=50)[0]["generated_text"]
    print("Bot:", output)
