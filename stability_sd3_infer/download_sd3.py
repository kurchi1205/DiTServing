import requests

def download_file_with_token(url, access_token, output_file_path):
    """
    Downloads a file from a URL that requires an access token.

    :param url: str, URL of the file to download
    :param access_token: str, access token for authorization
    :param output_file_path: str, path to save the downloaded file
    """
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()  # Raise an error for HTTP errors

        with open(output_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"File downloaded successfully and saved to {output_file_path}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

# Example usage
# url = "https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/text_encoders/clip_g.safetensors"
# url = "https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/text_encoders/clip_l.safetensors"
url = "https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/text_encoders/t5xxl_fp16.safetensors"
access_token = ""
output_file_path = "models/t5xxl_fp16.safetensors"

download_file_with_token(url, access_token, output_file_path)
