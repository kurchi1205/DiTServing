import argparse
from datetime import datetime
import time
import requests
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", type=str, required=True, help="Torchserve inference endpoint"
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Prompt for image generation"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="output-{}.jpg".format(str(datetime.now().strftime("%Y%m%d%H%M%S"))),
        help="Filename of output image",
    )
    return parser

if __name__=="__main__":
    args = get_args().parse_args()
    st_time = time.time()
    response = requests.post(args.url, data=args.prompt)
    print("Time taken: ", time.time() - st_time)

# python query.py --url http://localhost:8080/predictions/sd3-model/ --prompt "A beautiful girl"
# python query.py --url http://localhost:8080/predictions/sd3-model-batched/ --prompt "A beautiful girl"