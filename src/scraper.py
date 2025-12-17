import requests
from bs4 import BeautifulSoup
import json
import time
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://www.shl.com"
START_URL = "https://www.shl.com/solutions/products/product-catalog/"
DATA_DIR = "data/raw"
CATALOG_FILE = os.path.join(DATA_DIR, "catalog_links.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "assessments.json")
INTERMEDIATE_FILE = os.path.join(DATA_DIR, "assessments_partial.jsonl")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5"
}

def get_soup(session, url):
    for attempt in range(4):
        try:
            response = session.get(url, headers=HEADERS, timeout=60, verify=False)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed for {url}: {e}")
            time.sleep(5 * (attempt + 1))
    logging.error(f"Failed to fetch {url} after retries")
    return None

def fetch_catalog_page(session, start):
    current_url = f"{START_URL}?start={start}"
    logging.info(f"Scraping catalog offset {start}") 
    soup = get_soup(session, current_url)
    if not soup:
        return []

    links = soup.find_all('a', href=True)
    page_assessments = []
    
    for link in links:
        href = link['href']
        if "/solutions/products/product-catalog/view/" in href or "/products/product-catalog/view/" in href:
            full_url = BASE_URL + href if href.startswith("/") else href
            
            title = link.get_text(strip=True)
            if not title:
                title = " ".join([c.get_text(strip=True) for c in link.find_all(recursive=False)])
            
            page_assessments.append({
                "name": title,
                "url": full_url,
                "description": "",
                "type": "Unknown"
            })
    return page_assessments

def scrape_catalog():
    if os.path.exists(CATALOG_FILE):
        logging.info("Loading existing catalog links...")
        try:
            with open(CATALOG_FILE, 'r') as f:
                return json.load(f)
        except:
            logging.warning("Failed to load catalog file, rescraping.")

    assessments = []
    session = requests.Session()
    
    # Loop from start=0 up to 450 with step 12
    offsets = range(0, 450, 12)
    
    logging.info("Fetching catalog pages in parallel...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_offset = {executor.submit(fetch_catalog_page, session, start): start for start in offsets}
        for future in as_completed(future_to_offset):
            try:
                data = future.result()
                assessments.extend(data)
            except Exception as e:
                logging.error(f"Page failed: {e}")

    session.close()
    
    # Deduplicate
    unique_assessments = {a['url']: a for a in assessments}.values()
    assessments = list(unique_assessments)
    
    # Save intermediate
    with open(CATALOG_FILE, 'w') as f:
        json.dump(assessments, f, indent=2)
        
    return assessments

def scrape_details(assessment):
    url = assessment['url']
    session = requests.Session()
    soup = get_soup(session, url)
    if not soup:
        return assessment

    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc:
        assessment['description'] = meta_desc.get('content', '').strip()
    
    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
    if main_content:
        paragraphs = main_content.find_all('p')
        text_content = " ".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
        if text_content:
            if len(assessment['description']) < 50:
                 assessment['description'] = text_content
            else:
                 assessment['description'] += " " + text_content[:500]
    
    text_for_type = (assessment['name'] + " " + assessment['description']).lower()
    
    if any(k in text_for_type for k in ['java', 'python', 'coding', 'developer', 'engineer', 'c++', 'sql', 'technical', 'react', 'node', 'aws']):
        assessment['type'] = 'Technical'
    elif any(k in text_for_type for k in ['personality', 'behavior', 'preference', 'motivation', 'opq', 'cognitive', 'culture', 'sales', 'scenario']):
        assessment['type'] = 'Behavioral/Personality'
    elif any(k in text_for_type for k in ['numerical', 'verbal', 'deductive', 'reasoning', 'ability', 'general ability']):
         assessment['type'] = 'Cognitive'
    else:
        assessment['type'] = 'General/Skills'

    return assessment

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    logging.info("Starting Scraper Loop...")
    assessments = scrape_catalog()
    logging.info(f"Unique assessments: {len(assessments)}")

    assessments.sort(key=lambda x: x['url'])
    
    # Check for existing work
    processed_urls = set()
    if os.path.exists(INTERMEDIATE_FILE):
         with open(INTERMEDIATE_FILE, 'r') as f:
             for line in f:
                 try:
                     item = json.loads(line)
                     processed_urls.add(item['url'])
                 except: pass
    logging.info(f"Resuming... {len(processed_urls)} already processed.")

    remaining_assessments = [a for a in assessments if a['url'] not in processed_urls]
    
    logging.info(f"Starting Details Scrape for {len(remaining_assessments)} items...")
    
    # Open file in append mode 
    with open(INTERMEDIATE_FILE, 'a') as f_out:
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_assessment = {executor.submit(scrape_details, a.copy()): a for a in remaining_assessments}
            for i, future in enumerate(as_completed(future_to_assessment)):
                try:
                    data = future.result()
                    # Write immediately
                    f_out.write(json.dumps(data) + "\n")
                    f_out.flush() # Ensure it hits disk
                    
                    if i > 0 and i % 10 == 0:
                        logging.info(f"Processed +{i}")
                except Exception as e:
                    logging.error(f"Error processing item: {e}")
    
    # Finalize: Compile JSONL to JSON
    logging.info("Compiling final JSON...")
    final_data = []
    if os.path.exists(INTERMEDIATE_FILE):
        with open(INTERMEDIATE_FILE, 'r') as f:
            for line in f:
                try:
                    final_data.append(json.loads(line))
                except: pass
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    logging.info(f"Complete. Saved {len(final_data)} items to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
