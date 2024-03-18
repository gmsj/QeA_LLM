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
    question_elem = section.xpath('.//h3')[0]
    answer_elem = section.xpath('.//span')[0]

    info = {
        'question': get_elem_text(question_elem),
        'answer': get_elem_text(answer_elem)
    }

    table_items.append(info)
    
    return table_items

# Via requests o encoding ficou bizarro
def extract_info(page_source):
    full_results = []
    root = html.fromstring(page_source)
    articles = root.xpath("//li[contains(@class, 'List__ListItem-sc')]")
    for article in articles:
        results = get_question_and_answer(article)
        full_results.extend(results)

    return full_results