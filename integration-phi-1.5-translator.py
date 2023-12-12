import json
import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModel

def load_model(model_name):
    print('Loading model...')
    model = AutoModel.from_pretrained(model_name)
    return model

def load_tokenizer(model_name):
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def load_document(filepath):
    print('Loading document...')
    file = open(filepath, 'r', encoding='utf-8')
    document = json.loads(file.read())
    file.close()

    return document

def load_QA_database(model, tokenizer, document):
    print('Tokenizing data...')
    question_data = []
    answer_data = []

    for item in document:
        question_data.append(item["question"])
        answer_data.append(item["answer"])

    tokenized_questions = [tokenizer(question, return_tensors="pt") for question in question_data]

    embeddings = []
    for tokens in tokenized_questions:
        with torch.no_grad():
            output = model(**tokens)
        embeddings.append(output.last_hidden_state.mean(dim=1).squeeze().numpy())

    embeddings = np.vstack(embeddings)
    return embeddings, question_data, answer_data

def index_into_faiss(embeddings):
    print('Indexing database in faiss...')
    vector_dimension = embeddings.shape[1]  
    index = faiss.IndexFlatL2(vector_dimension)
    faiss.normalize_L2(embeddings) 
    index.add(embeddings)
    return index

def search_into_faiss(model, tokenizer, query, index, max_results):
    query_tokens = tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        query_embedding = model(**query_tokens).last_hidden_state.mean(dim=1).squeeze().numpy()
    distance, nearest_neighbours = index.search(np.array([query_embedding]), max_results)

    return distance, nearest_neighbours

def get_data_neighbours(question_data, answer_data, nearest_neighbours):
    
    similar_queries = []
    context_queries = []

    for i, idx in enumerate(nearest_neighbours[0]):
        similar_queries.append(question_data[idx])
        context_queries.append(answer_data[idx])

    return similar_queries, context_queries

def generate_answer(model, tokenizer, input_text, max_length=2048):
    print('Generating answer...')
    inputs = tokenizer(input_text, return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs, max_length=max_length)
    text = tokenizer.batch_decode(outputs)[0]
    return text

def load_llm(model_name):
    print('Loading LLM model...')
    # torch.set_default_device('cuda')
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    return model, tokenizer

def load_translate_model(model_name):
    print('Loading Translator model...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return model, tokenizer

def translate_sentences(model, tokenizer, text_to_translate):
    input_ids = tokenizer.encode(text_to_translate, return_tensors="pt")

    max_length = 150
    num_beams = 5
    temperature = 0.8
    top_k = 50
    top_p = 0.95
    no_repeat_ngram_size = 2

    # Por causa do top_p e temperature
    do_sample = True

    translation_ids = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=num_beams,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=do_sample
    )
    translated_text = tokenizer.decode(translation_ids[0], skip_special_tokens=True)
    return translated_text

model_name = 'microsoft/MiniLM-L12-H384-uncased'
llm_model_name = 'microsoft/phi-1_5'
document = load_document('./documents/nubank_4uersz.html.json')
model = load_model(model_name)
tokenizer = load_tokenizer(model_name)
embeddings, question_data, answer_data = load_QA_database(model, tokenizer, document)
faiss_index = index_into_faiss(embeddings)

llm_model, llm_tokenizer = load_llm(llm_model_name)

while True:
    print()
    
    input_text = input('Type your question: ')
    
    if input_text == 'exit':
        break
    
    distance, nearest_neighbours = search_into_faiss(model, tokenizer, input_text, faiss_index, 5)
    similar_queries, context_queries = get_data_neighbours(question_data, answer_data, nearest_neighbours)

    # print('\nResults:', '\n\n', '\n\n'.join(similar_queries))
    # print('\n-------------------------------------------------------------\n')
    # print('Context answers:', '\n\n'.join(context_queries))

    input_context = '\n'.join(context_queries)

    translate_model, translate_tokenizer = load_translate_model('unicamp-dl/translation-pt-en-t5')
    input_text = translate_sentences(translate_model, translate_tokenizer, input_text)
    input_context = translate_sentences(translate_model, translate_tokenizer, input_context)

    del translate_model, translate_tokenizer

    template = f"You are an AI assistant and you will analyze the context given below and based on this informed context, you will answer the question in a summarized and clear way.\n\nContext: {input_context}\n\nQuestion: {input_text}\n\nAnswer:"
    # print(f"\n\n\n{template}\n\n\n")
    result = generate_answer(llm_model, llm_tokenizer, template)
    raw_answer = result[result.index('Answer:'):]

    translate_model, translate_tokenizer = load_translate_model('unicamp-dl/translation-en-pt-t5')
    final_answer = translate_sentences(translate_model, translate_tokenizer, raw_answer)
    print(final_answer)

    del translate_model, translate_tokenizer