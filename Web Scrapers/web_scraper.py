import os
import time
import json
import string
import random
import codecs
import requests
from selenium import webdriver

import scraper_caixa as caixa
import scraper_nubank as nubank
import scraper_mastercard as mastercard
import scraper_ferreiracosta as ferreiracosta

##### constants #####

# chromedriver_path = './chromedriver'
downloaded_pages_path = './examples'

global random_id
random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.111 Safari/537.36',
    ]

def remove_unwanted_terms(raw_text, unwanted_terms):
    for term in unwanted_terms:
        raw_text = raw_text.replace(term, '')
    return raw_text

def extract_domain_from_url(raw_url):
    unwanted_terms =['https://', 'http://', 'www.']
    raw_url = remove_unwanted_terms(raw_url, unwanted_terms)
    domain = raw_url.split('.')[0]
    return domain

def download_pages(urls_list, method):
    # Talvez remover essa variável como global
    global random_id
    
    for item in urls_list:
        random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        domain = extract_domain_from_url(item)
        print('Getting page:', item)
        if method == 'selenium':
            get_page_with_selenium(item, domain)
        elif method == 'requests':
            get_page_with_requests(item, domain)
        else:
            raise Exception('The method provided is not supported')
        
        print('Waiting time between requests')
        print()
        time.sleep(10)

def get_page_with_selenium(url, domain):
    
    # user_agent = random.choice(user_agents)
    # options = webdriver.ChromeOptions()
    # options.add_argument("--headless")
    # options.add_argument(f"user-agent={user_agent}")
    # driver = webdriver.Chrome(executable_path=chromedriver_path)

    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(options=options)
    
    driver.implicitly_wait(0.5)
    driver.get(url)
    page_source = driver.page_source

    if not domain:
        return page_source

    filepath = os.path.join(f'{downloaded_pages_path}/{domain}_{random_id}.html')
    file = codecs.open(filepath, "w", "utf-8")
    file.write(page_source)
    driver.quit()
    return page_source

def get_page_with_requests(url, domain = None):
    user_agent = random.choice(user_agents)
    print(f'Resquesting with [{user_agent}]')
    headers = {'User-Agent': user_agent}
    response = requests.get(url, 'html.parser', headers=headers)
    page_source = response.text

    if response.status_code != 200:
        print(f'URL is Not reachable, status_code: {response.status_code}')
    
    if not domain:
        return page_source

    filepath = os.path.join(f'{downloaded_pages_path}/{domain}_{random_id}.html')
    file = codecs.open(filepath, "w", "utf-8")
    file.write(page_source)
    return page_source

def list_files_recursively(directory_path):
    files_list = []
    for current_folder, subfolders, files in os.walk(directory_path):
        for file in files:
            full_path = os.path.join(current_folder, file)
            files_list.append(full_path)
    return files_list

def get_offline_page(filepath):
    filepath = os.path.join(filepath)
    file = codecs.open(filepath, "r", "utf-8")
    return file.read()

def save_result(filename, content):
    filepath = os.path.join(f'./results/{filename}.json')
    file = codecs.open(filepath, "w", "utf-8")
    content = json.dumps(content, indent = 4, ensure_ascii=False)
    file.write(content)

def offline_extractor(file_path):
    print('Extracting info from:', file_path)
    page_source = get_offline_page(file_path)

    if 'caixa' in file_path:
        result = caixa.extract_info(page_source)
    elif 'ferreiracosta' in file_path:
        result = ferreiracosta.extract_info(page_source)
    elif 'mastercard' in file_path:
        result = mastercard.extract_info(page_source)
    elif 'nubank' in file_path:
        result = nubank.extract_info(page_source)
    else:
        print('Filename is not supported')
        exit()
    
    filename = file_path.split('/')[-1]
    save_result(filename, result)
    
    return result