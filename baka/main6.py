# Step 1: Install the 'transformers' library from Hugging Face
# This is like getting your chef's tools ready.
#!pip install transformers//only use when in colab or jupyter notebook

# Step 2: Import the 'pipeline' tool
# The pipeline is like an expert assistant that handles the complicated steps.
from transformers import pipeline

# Step 3: Hire your junior chef
# We're loading a pre-trained model (a smaller version of GPT-2).
# This chef already read a lot of books and is ready for your instructions!
text_generator = pipeline("text-generation", model="gpt2")

# Step 4: Give your instruction (the "prompt")
# This is the beginning of the recipe you want the chef to finish.
prompt = "In a world where robots can dream, they often dream of"

# Step 5: Let the chef create!
# The AI will now use its 'attention' power to predict what comes next.
generated_text = text_generator(prompt, max_length=25, num_return_sequences=1)

# Step 6: See the result
print(generated_text[0]['generated_text'])