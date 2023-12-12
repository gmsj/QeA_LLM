import json
import faiss
import torch
import numpy as np

from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModel

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

def generate_answer(model, input_text, input_context, prompt):
    print('Generating answer...')
    chain = LLMChain(prompt=prompt, llm=model, output_key="answer")
    result = chain({"context": input_text, "input": input_context})
    return result

def load_llm(model_name):
    print('Loading LLM model...')
    model_kwargs = {"temperature": 0.1, "max_new_tokens": 200, "repetition_penalty": 2}
    api_token = "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"

    model = HuggingFaceHub(repo_id=model_name, model_kwargs=model_kwargs, huggingfacehub_api_token=api_token)

    return model

model_name = 'microsoft/MiniLM-L12-H384-uncased'
llm_model_name = 'tiiuae/falcon-7b-instruct'
document = load_document('./documents/caixa_ummipx.html.json')
model = load_model(model_name)
tokenizer = load_tokenizer(model_name)
embeddings, question_data, answer_data = load_QA_database(model, tokenizer, document)
faiss_index = index_into_faiss(embeddings)

llm_model = load_llm(llm_model_name)

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

    template = f"Você é um assistente de IA e analisará o contexto fornecido a seguir e com base nesse contexto informado, responderá à pergunta de forma resumida e clara.\n\nContexto: {input_context}\n\nPergunta: {input_text}\n\nResposta:"
    prompt = PromptTemplate(template=template.lower(), input_variables=["context", "input"])

    result = generate_answer(llm_model, input_text, input_context, prompt)
    print(result['answer'])