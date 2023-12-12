from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = 'unicamp-dl/translation-en-pt-t5'
model_name = 'unicamp-dl/translation-pt-en-t5'

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Example text in Portuguese to translate
text_to_translate = "Insira aqui o texto em português que você deseja traduzir."

# Tokenize the input text
input_ids = tokenizer.encode(text_to_translate, return_tensors="pt")

# Generate translation with max length parameter
max_length = 1024  # You can adjust this value based on your preference
translation_ids = model.generate(input_ids, max_length=max_length)

# Decode the generated translation IDs
translated_text = tokenizer.decode(translation_ids[0], skip_special_tokens=True)

# Print the translated text
print("Texto original :", text_to_translate)
print("Texto traduzido:", translated_text)
