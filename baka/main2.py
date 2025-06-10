from transformers import pipeline

#Using pre tarained model for chatbot
generator = pipeline("text-generation", model ='gpt2')

#Making a chatbot function
promt = "you are a poetry chatbot, write a poem about the moon"
answer = generator(promt, max_length=50, num_return_sequences=1)

print(answer[0]['generated_text'])