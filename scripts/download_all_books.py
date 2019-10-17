"""
Download all zip files of "日本古典籍データセット" from CODH home page.

url: http://codh.rois.ac.jp/
"""


import argparse
import json
from pathlib import Path
import time
from typing import List
import zipfile

import requests
from bs4 import BeautifulSoup


BASE_URL = 'http://codh.rois.ac.jp'
BOOKS_URL = 'http://codh.rois.ac.jp/pmjt/book/'


def get_book_urls() -> List[str]:
    """Get each book page urls from book list page."""
    with requests.get(BOOKS_URL) as res:
        res.encoding = res.apparent_encoding
        content = res.text
    soup = BeautifulSoup(content, features="html.parser")
    table = soup.find(id='list')
    tbody = table.find('tbody')
    books = tbody.find_all('tr')
    book_urls = [BOOKS_URL + b.find('a')['href'] for b in books]
    return book_urls


def get_book_data(book_url: str) -> dict:
    """Get zip file url and license text from the book page."""
    with requests.get(book_url) as res:
        res.encoding = res.apparent_encoding
        content = res.text
    soup = BeautifulSoup(content, features="html.parser")
    links_in_article = soup.find('article').find_all('a')
    zip_url = None
    license = None
    for link in links_in_article:
        if zip_url is None and link['href'].endswith('.zip'):
            zip_url = BASE_URL + link['href']
        if link.get('rel', None) == ['license']:
            license = link.text

    return {'book_url': book_url,
            'zip_url': zip_url,
            'license_text': license}


def download_zip(zip_url: str, save_path: Path) -> None:
    """Download zip file."""
    with requests.get(zip_url, stream=True) as res:
        with open(save_path, 'wb') as f:
            for chunk in res.iter_content(chunk_size=512):
                if chunk:
                    f.write(chunk)


def main():
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', type=str)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # get all book urls
    book_urls = get_book_urls()

    # get all zip file urls
    all_book_data = []
    for url in book_urls:
        all_book_data.append(get_book_data(url))
        time.sleep(0.1)
    metadata_path = save_dir / 'metadata.json'
    with metadata_path.open('w') as f:
        json.dump({'data': all_book_data}, f, indent=2)

    # download and extract all zip files
    download_dir = save_dir / 'zipfiles'
    output_dir = save_dir / 'books'

    download_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    for i, book_data in enumerate(all_book_data):
        print('[{}/{}] {}'.format(i + 1, len(all_book_data), book_data['book_url']))
        book_id = book_data['book_url'].split('/')[-2]
        zip_path = download_dir / (book_id + '.zip')
        download_zip(book_data['zip_url'], zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(output_dir)


if __name__ == '__main__':
    main()
