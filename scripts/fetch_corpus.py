"""
Downloads the document corpus from the URL list I curated 
with the health policy related public documents from sources
like CMS, HHS, WHO, etc. And write those files into the data/raw/
folder.
"""

import csv
from pathlib import Path
import time
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
corpus_list = "document-list.csv"

# Status codes worth retrying: rate limiting and server-side hiccups.
# A 403 from bot protection won't fix itself on retry, so it's deliberately
# excluded here and left to raise_for_status() below.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


# The fetch function writes the content of a file to the local computer
# The function takes 3 parameters - the full URL of the file, the name of
# the file, and the requests Session to fetch it with
def fetch(url: str, dest_filename: str, session: requests.Session, max_retries: int = 3) -> bool:

    """Download a single specified file to the destination directory"""

    # The timeout parameter indicates the program will wait 15 seconds
    # to connect to the source, and get content, if it takes more than
    # 15 seconds, the code will raise a requests.exceptions.Timeout
    timeout = 15

    dest_filepath = RAW_DIR / dest_filename
    if dest_filepath.exists():
        return False

    # Retry on likely-transient failures, waiting longer between each
    # attempt (2s, 4s, 8s, ...). Any other status just falls through once.
    for attempt in range(1, max_retries + 1):
        response = session.get(url, timeout=timeout)

        if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
            wait = 2 ** attempt
            print(f"Got {response.status_code} for {dest_filename}, "
                  f"retrying in {wait}s (attempt {attempt}/{max_retries})")
            time.sleep(wait)
            continue

        break

    # Raise HTTPError for bad responses (4xx, 5xx)
    response.raise_for_status()

    # Write the content of the file to the destination directory
    (dest_filepath).write_bytes(response.content)
    return True


# The main function
def main():

    corpus_filepath = DATA_DIR / corpus_list
    if corpus_filepath.exists():

        # A single Session reuses the underlying connection and carries any
        # cookies the server sets across requests, instead of every fetch()
        # call looking like a brand new, unrelated visitor to the site.
        session = requests.Session()

        # The User-Agent header will make this app pretend to be a web browser
        # making the request to the web server hosting the desired file. Without
        # this, some servers may not entertain the request. Set once on the
        # session so it applies to every request made with it.
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0"
        })

        # Collect (filename, error) pairs so failures are summarized at the
        # end instead of only scrolling by in the per-file console output.
        failures = []

        # Read the content of the corpus file line-by-line
        with open((DATA_DIR / corpus_list), newline="", encoding="utf-8") as f:
            content = csv.DictReader(f)

            # Read each line from the CSV file to get the filename and the url
            for i, line in enumerate(content, start=1):

                # Adding a 1 second delay between 2 fetch, to reduce the chance
                # of getting '403 Client Error: Forbidden for url'
                time.sleep(1)

                try:
                    if fetch(line["url"], line["local_filename"], session):
                        print(f"Wrote file {i} name {line['local_filename']} in the destination folder")
                    else:
                        print(f"File {i} name {line['local_filename']} already exists in the destination folder")

                except (requests.exceptions.RequestException, OSError) as e:
                    print(f"Failed to fetch {line['local_filename']}: {e}")
                    failures.append((line["local_filename"], str(e)))

        if failures:
            print(f"\n{len(failures)} file(s) failed to download:")
            for filename, error in failures:
                print(f"  - {filename}: {error}")

    else:
        print("The corpus file not found")

if __name__ == "__main__":
    main()
