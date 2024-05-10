import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build

def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res['items']


def extract_text_from_p_tags(url, header):
    response = requests.get(url, header)
    if response.status_code == 200:

        soup = BeautifulSoup(response.text, 'html.parser')

        for element in soup(['script', 'style', 'template','a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            element.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

        text = '\n'.join(chunk for chunk in chunks if len(chunk.split())>3)
        return text
    else:
        return "Failed to retrieve the webpage."