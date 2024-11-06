import os
import re
import uuid
import zipfile
from typing import List, Union
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

import backoff
import requests
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from requests import RequestException

# User-Agent and Authorization handling
USER_AGENT = "runpod-python/0.0.0 (https://runpod.io; support@runpod.io)"

def get_auth_header() -> dict:
    """
    Produce a header dict with the `Authorization` key derived from
    credentials.get("api_key") OR os.getenv('RUNPOD_AI_API_KEY')
    """
    auth = os.getenv("RUNPOD_AI_API_KEY", "")
    return {
        "Content-Type": "application/json",
        "Authorization": auth,
        "User-Agent": USER_AGENT,
    }


# Helper functions
def calculate_chunk_size(file_size: int) -> int:
    """
    Calculates the chunk size based on the file size.
    """
    if file_size <= 1024 * 1024:  # 1 MB
        return 1024  # 1 KB
    if file_size <= 1024 * 1024 * 1024:  # 1 GB
        return 1024 * 1024  # 1 MB
    return 1024 * 1024 * 10  # 10 MB


# HTTP Client Sessions
class SyncClientSession(requests.Session):
    """
    Inherits requests.Session to override `request()` method for tracing.
    """

    def request(self, method, url, **kwargs):
        """
        Override for tracing. Captures metrics for connection and transfer times.
        """
        request_kwargs = {k: v for k, v in kwargs.items() if k in requests.Request.__init__.__code__.co_varnames}
        send_kwargs = {k: v for k, v in kwargs.items() if k not in request_kwargs}

        req = requests.Request(method, url, **request_kwargs)
        prepped = self.prepare_request(req)

        settings = self.merge_environment_settings(
            prepped.url,
            send_kwargs.get("proxies"),
            send_kwargs.get("stream"),
            send_kwargs.get("verify"),
            send_kwargs.get("cert"),
        )
        send_kwargs.update(settings)

        response = self.send(prepped, **send_kwargs)
        return response


def AsyncClientSession(*args, **kwargs):
    """
    Factory method for creating an aiohttp.ClientSession with custom headers and a tracer.
    """
    return ClientSession(
        connector=TCPConnector(limit=0),
        headers=get_auth_header(),
        timeout=ClientTimeout(600, ceil_threshold=400),
        *args,
        **kwargs,
    )


# Main functionality
@backoff.on_exception(backoff.expo, RequestException, max_tries=3)
def download_file(url: str, path_to_save: str) -> str:
    """
    Downloads a single file from the given URL and saves it to the specified path.
    Returns the file extension.
    """
    with SyncClientSession().get(url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=30) as response:
        response.raise_for_status()
        content_disposition = response.headers.get('Content-Disposition')
        file_extension = ''
        file_extension = os.path.splitext(urlparse(url).path)[1]
        file_size = int(response.headers.get('Content-Length', 0))
        chunk_size = calculate_chunk_size(file_size)
        with open(path_to_save + file_extension, 'wb') as file_path:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file_path.write(chunk)
        return file_extension


def download_files_from_urls(job_id: str, urls: Union[str, List[str]]) -> List[str]:
    """
    Downloads files from a single URL or a list of URLs, saving them to a job-specific directory.
    Returns a list of absolute file paths for the downloaded files.
    """
    download_directory = os.path.abspath(os.path.join('jobs', job_id, 'downloaded_files'))
    os.makedirs(download_directory, exist_ok=True)

    def download_file_to_path(url: str) -> str:
        if not url:
            return None

        file_name = f'{uuid.uuid4()}'
        output_file_path = os.path.join(download_directory, file_name)

        try:
            file_extension = download_file(url, output_file_path)
        except RequestException as err:
            print(f"Failed to download {url}: {err}")
            return None

        return os.path.abspath(f"{output_file_path}{file_extension}")

    if isinstance(urls, str):
        urls = [urls]

    with ThreadPoolExecutor() as executor:
        downloaded_files = list(executor.map(download_file_to_path, urls))

    return [file for file in downloaded_files if file]


def file(file_url: str) -> dict:
    """
    Downloads a single file from a given URL, assigns it a random name, and handles zip file extraction.
    Returns an object containing:
    - The absolute path to the downloaded file
    - File type
    - Original file name
    - Extracted path (if the file was a zip)
    """
    os.makedirs('job_files', exist_ok=True)

    download_response = SyncClientSession().get(file_url, headers={"User-Agent": USER_AGENT}, timeout=30)
    download_response.raise_for_status()

    original_file_name = re.findall(r'filename="(.+)"', download_response.headers.get("Content-Disposition", "")) or \
                         [os.path.basename(urlparse(file_url).path)]
    original_file_name = original_file_name[0]
    file_type = os.path.splitext(original_file_name)[1].replace('.', '')

    file_name = f'{uuid.uuid4()}'
    output_file_path = os.path.join('job_files', f'{file_name}.{file_type}')

    with open(output_file_path, 'wb') as output_file:
        output_file.write(download_response.content)

    unziped_directory = None
    if file_type == 'zip':
        unziped_directory = os.path.join('job_files', file_name)
        os.makedirs(unziped_directory, exist_ok=True)
        with zipfile.ZipFile(output_file_path, 'r') as zip_ref:
            zip_ref.extractall(unziped_directory)
        unziped_directory = os.path.abspath(unziped_directory)

    return {
        "file_path": os.path.abspath(output_file_path),
        "type": file_type,
        "original_name": original_file_name,
        "extracted_path": unziped_directory
    }
