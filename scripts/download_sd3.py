import os
import requests
import argparse
from pathlib import Path


def download_file_with_token(url: str, access_token: str, output_file_path: str) -> None:
    """
    Downloads a file from a URL that requires an access token.

    Args:
        url (str): URL of the file to download
        access_token (str): Access token for authorization
        output_file_path (str): Path to save the downloaded file
    """
    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        # Create directory structure if it doesn't exist
        output_path = Path(output_file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with requests.get(url, headers=headers, stream=True) as response:
            response.raise_for_status()  # Raise an error for HTTP errors
            with open(output_file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        print(f"File downloaded successfully and saved to {output_file_path}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download files with access token")
    parser.add_argument("-t", "--access-token", help="Access token for authorization", required=True)
    parser.add_argument("-u", "--url", help="URL of the file to download", required=True)
    parser.add_argument("-o", "--output-file", help="Path to save the downloaded file", required=True)

    args = parser.parse_args()

    download_file_with_token(args.url, args.access_token, args.output_file)

if __name__ == "__main__":
    main()