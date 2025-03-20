import asyncio
import json
import os
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict, Union, Tuple
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig, LLMExtractionStrategy
from urllib.parse import urlparse, quote
from googlesearch import search

# Load environment variables (if any)
load_dotenv()

# Configuration loaded from .env or config.json
config = {
    "sources": ["forbesindia.com"],
    "keywords": ["Corporate India Risk Index Framework", "Corporate India Risk Dimensions"],
    "country": "India"
}

# Pydantic Models (unchanged)
class RepoRateData(BaseModel):
    date: str = Field(..., description="Date of the repo rate announcement")
    repo_rate: float = Field(..., description="Repo rate percentage")

class ContentMetadata(BaseModel):
    content_type: str = Field(..., description="Type of content (text, table, image, pdf, etc)")
    location: str = Field(..., description="Location/section where content appears in document")
    extracted_text: Optional[str] = Field(None, description="Extracted text content if available")
    table_data: Optional[List[List[str]]] = Field(None, description="Table data if content is present tabular")
    imagedata: Optional[List[str]] = Field(None, description="List of image urls content if present")
    pdf_metadata: Optional[Dict] = Field(None, description="PDF-specific metadata if content is PDF")
    videodata: Optional[List[str]] = Field(None, description="List of video urls content if present")

class NumericData(BaseModel):
    value: str = Field(..., description="Actual numerical value")
    data_type: str = Field(..., description="Type of numeric data (percentage, ratio, year, currency, etc.)")
    context: str = Field(..., description="Contextual description of the numeric value")
    unit: Optional[str] = Field(None, description="Measurement unit if applicable")
    source: Optional[str] = Field(None, description="Source of the numerical data")
    time_period: Optional[str] = Field(None, description="Relevant time period for the data")
    comparison: Optional[Dict] = Field(None, description="Comparative data if available")
    report_reference: Optional[str] = Field(None, description="Associated report/document reference")

class RiskMetrics(BaseModel):
    risk_level: str = Field(..., description="Risk level assessment (Low/Medium/High)")
    impact_score: float = Field(..., description="Numerical score indicating potential impact (0-10)")
    likelihood_score: float = Field(..., description="Numerical score indicating likelihood (0-10)")
    trend: str = Field(..., description="Current trend of risk (Increasing/Stable/Decreasing)")
    mitigation_measures: List[str] = Field(..., description="List of risk mitigation measures")
    description: str = Field(..., description="All the content related to it")
    related_content: List[ContentMetadata] = Field(..., description="Content pieces discussing this risk")

class FinalDataSchema(BaseModel):
    title: str = Field(..., description="Document title with edition/year")
    effective_date: datetime = Field(..., description="Effective date of analysis")
    valid_until: Optional[datetime] = Field(None, description="Expiry date of analysis")
    source: str = Field(..., description="Publisher/source organization")
    document_url: str = Field(..., description="Official document URL")
    geographic_coverage: List[str] = Field(..., description="Covered states/regions")
    about: str = Field(..., description="Brief summary of the document content")
    description: str = Field(..., description="Detailed description of the document who stated what")
    report_period: Dict[str, Union[int, str]] = Field(..., description="Analysis period (start_year, end_year, fiscal_year)")
    key_metrics: Dict[str, NumericData] = Field(..., description="Critical numerical indicators (GDP, inflation, etc.)")
    reports_cited: List[str] = Field(None, description="List of referenced official documents")
    imagedata: Optional[List[str]] = Field(None, description="List of image urls content if present")
    videodata: Optional[List[str]] = Field(None, description="List of video urls content if present")
    pdf_metadata: Optional[Dict] = Field(None, description="PDF-specific metadata if content is PDF")


# Get LLM extraction strategy
def get_llm_strategy() -> LLMExtractionStrategy:
    return LLMExtractionStrategy(
        provider="openai/gpt-4o",
        api_token=os.getenv("OPENAI_API_KEY"),
        schema=FinalDataSchema.model_json_schema(),
        extraction_type="schema",
        instruction="Extract all {FinalDataSchema} data from the page.  Be sure to capture all image, video, and PDF URLs if available. If 'pdf_metadata' is present, it should contain a list of 'urls'.",
        input_format="markdown",
        verbose=True,
    )

# Convert URL to safe filename
def get_safe_filename(url: str) -> str:
    """Convert URL to safe filename."""
    return "".join(c if c.isalnum() else "_" for c in url).rstrip("_") + ".json"

# Save extracted data to a local file
def save_data_locally(data, site, keyword, url):
    local_folder = f"output/{site}/{keyword.replace(' ', '_')}"
    os.makedirs(local_folder, exist_ok=True)
    file_name = get_safe_filename(url)
    local_path = os.path.join(local_folder, file_name)
    try:
        with open(local_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Saved data locally to {local_path}")
    except Exception as e:
        print(f"Error saving data locally: {e}")

# Asynchronously download media files
async def download_media_async(media_url, local_folder, media_type, media_name):
    """Asynchronously download media files (image, video, pdf)."""
    try:
        # URL-encode the URL to handle special characters
        encoded_media_url = quote(media_url, safe='/:')
        print(f"Attempting to download: {encoded_media_url}")

        response = await asyncio.to_thread(requests.get, encoded_media_url, stream=True, timeout=30)  # Add timeout

        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        file_extension = media_url.split('.')[-1]
        local_media_path = os.path.join(local_folder, f"{media_name}.{file_extension}")

        with open(local_media_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

        print(f"Downloaded and saved {media_type}: {media_name}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {media_type} {media_url}: {e}")
        return False  # Indicate failure
    except Exception as e:
        print(f"Error saving {media_type} {media_url}: {e}")
        return False  # Indicate failure
    return True # Indicate success

# Save media files (images, videos, pdfs) asynchronously
async def save_media_files_async(data, site, keyword, url):
    """Asynchronously save images, videos, and PDFs."""
    local_folder = f"output/{site}/{keyword.replace(' ', '_')}"
    os.makedirs(local_folder, exist_ok=True)

    media_folders = {
        'images': os.path.join(local_folder, 'images'),
        'videos': os.path.join(local_folder, 'videos'),
        'pdfs': os.path.join(local_folder, 'pdfs')
    }

    for folder in media_folders.values():
        os.makedirs(folder, exist_ok=True)

    tasks = []
    downloaded_all = True  # Track if all media were downloaded successfully

    if 'imagedata' in data and data['imagedata']:
        for idx, image_url in enumerate(data['imagedata']):
            download_success = await download_media_async(image_url, media_folders['images'], 'image', f"image_{idx + 1}")
            downloaded_all = downloaded_all and download_success

    if 'videodata' in data and data['videodata']:
        for idx, video_url in enumerate(data['videodata']):
            download_success = await download_media_async(video_url, media_folders['videos'], 'video', f"video_{idx + 1}")
            downloaded_all = downloaded_all and download_success

    if 'pdf_metadata' in data and data['pdf_metadata'] and 'urls' in data['pdf_metadata']:
        for idx, pdf_url in enumerate(data['pdf_metadata']['urls']):  # Access URLs inside pdf_metadata
            download_success = await download_media_async(pdf_url, media_folders['pdfs'], 'pdf', f"document_{idx + 1}")
            downloaded_all = downloaded_all and download_success

    # Wait for all tasks to complete (gather returns when all tasks are done)
    # We don't actually need the results, just the completion
    if tasks:
        await asyncio.gather(*tasks)

    return downloaded_all

# Fetch and process the data
async def fetch_and_process_data(crawler: AsyncWebCrawler, base_url: str, llm_strategy: LLMExtractionStrategy, session_id: str) -> Tuple[List[dict], bool]:
    try:
        result = await crawler.arun(
            url=base_url,
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                extraction_strategy=llm_strategy,
                css_selector="",
                session_id=session_id,
            ),
        )
        if not (result.success and result.extracted_content):
            print(f"Error fetching data: {result.error_message}")
            return [], False

        extracted_data = json.loads(result.extracted_content)
        if not extracted_data:
            print("No Data found")
            return [], False

        # Save media files
        all_media_downloaded = True
        for data in extracted_data:
            media_downloaded = await save_media_files_async(data, extract_domain(base_url), "some_keyword", base_url)
            all_media_downloaded = all_media_downloaded and media_downloaded

        if not all_media_downloaded:
            print("One or more media files failed to download.")

        # Clean extracted data
        for data in extracted_data:
            keys_to_remove = [key for key, value in data.items() if value is None]
            if data.get("error") is False and "error" not in keys_to_remove:
                keys_to_remove.append("error")
            for key in keys_to_remove:
                data.pop(key, None)

        print(f"Extracted {len(extracted_data)} items")
        return extracted_data, False
    except Exception as e:
        print(f"Error fetching data: {e}")
        return [], True

# Perform scraping tasks for a given keyword and site
async def scrapper(keyword, site, urls):
    llm_strategy = get_llm_strategy()
    session_id = "colab_session_123"
    print(f"URLs for keyword '{keyword}' on site {site}:")
    for url in urls:
        async with AsyncWebCrawler() as crawler:
            data, no_results_found = await fetch_and_process_data(crawler, url, llm_strategy, session_id)
            if no_results_found:
                print("No data found. Ending crawl.")
            elif data:
                # Apply filter for financial years (up to 2 years in the past)
                current_year = datetime.now().year
                valid_data = [item for item in data if 'report_period' in item and 'start_year' in item['report_period'] and
                              int(item['report_period']['start_year']) >= current_year - 2]
                if valid_data:
                    save_data_locally(valid_data, site, keyword, url)

# Generate URLs and scrape them
async def generate_url_json():
    final_result = {}
    for source in config["sources"]:
        final_result[source] = []
        for keyword in config["keywords"]:
            print(f"Processing '{keyword}' for '{source}'...")
            urls = search_urls(keyword, source)
            if urls:
                final_result[source].append({
                    "keyword": keyword,
                    "query": f"{keyword} site:{source} {config['country']}",
                    "urls": urls
                })
                await scrapper(keyword, source, urls)
    return final_result

# Perform the Google search for the URLs
def search_urls(keyword, site):
    query = f"{keyword} site:{site} {config['country']}"
    urls = list(search(query, num_results=10))
    return urls

def extract_domain(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

# Main function to execute the script
if __name__ == "__main__":
    asyncio.run(generate_url_json())
