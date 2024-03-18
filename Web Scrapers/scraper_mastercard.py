import json
from lxml import html
from unicodedata import normalize

def get_elem_text(elem, unwanted_chars = ['\n', '\t', '\r']):
    raw_text = elem.text_content()
    raw_text = raw_text.lower()
    for char in unwanted_chars:
        raw_text = raw_text.replace(char, '')
    text = ' '.join(raw_text.split())
    text = normalize('NFKD', text).encode('ASCII','ignore').decode('ASCII')
    text = ''.join(char for char in text if char.isalnum() or ' ')
    if text == '':
        return 'Nao encontrado'
    else:
        return text

def get_question_and_answer(section):
    table_items = []

    raw_text = get_elem_text(section)
    text_splited = raw_text.split('p: ')
    text_splited = list(filter(lambda x: x != '', text_splited))

    # A página tem umas coisas quebradas como:
    # P: Por que meu cartão está sendo upgraded para um cartão com chip?
    # P: E se eu quiser parar a cobrança de conta automática?

    for item in text_splited:
        tmp = item.split('r: ')
        if len(tmp) > 1:
            info = {
                'question': tmp[0],
                'answer': tmp[1]
            }
        else:
            info = {
                'question': tmp[0],
                'answer': 'Nao encontrado'
            }
        table_items.append(info)

    return table_items

def extract_info(page_source):
    full_results = []
    root = html.fromstring(page_source)
    articles = root.xpath("//div[contains(@class, 'text-editor text-article')]")
    for article in articles:
        results = get_question_and_answer(article)
        full_results.extend(results)

    return full_results