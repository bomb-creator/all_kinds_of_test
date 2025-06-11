# 1. Install the necessary libraries. 'sentencepiece' is often needed for translation models.
#!pip install transformers sentencepiece

# 2. Import the pipeline tool, our easy way to use models.
from transformers import pipeline

# 3. Load your 'Translation Team'.
# This model is specifically an English-to-French translation team.
# model="Helsinki-NLP/opus-mt-en-fr"
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

# 4. Write the English sentence you want the team to translate.
english_sentence = "The old library at the end of the street contains thousands of ancient books."

# 5. Give the sentence to your 'translator' team.
# BEHIND THE SCENES: The Encoder reads the English, and the Decoder writes the French.
french_translation = translator(english_sentence)

# 6. Show the result from your French agent!
print(french_translation[0]['translation_text'])