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

    # As duas barras são usadas para fazer uma busca em profundidade pois as vezes estão dentro de divs
    # Nos casos de "Não encontrado" ocorre que há uma parágrafo vazio
    titles = section.xpath('.//h5')
    contents = section.xpath('.//p')

    for i in range(0, len(titles)):
        info = {
            'question': get_elem_text(titles[i]),
            'answer': get_elem_text(contents[i])
        }
        table_items.append(info)
    
    return table_items

def extract_info(page_source):
    full_results = []
    root = html.fromstring(page_source)
    articles = root.xpath("//div[contains(@class, 'section-artigo with-box')]")
    for article in articles:
        # title = section = article.xpath(".//span[contains(@class, 'wp-menu')]")[0].text_content()
        section = article.xpath(".//div[contains(@class, 'colsm-7')]")[0]
        results = get_question_and_answer(section)
        full_results.extend(results)

    return full_results