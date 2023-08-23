import sys
import requests
from util import *
import pandas as pd


# Get video metadata and save to a file
def main(argv):
    vid_path = '/home/paperspace/git/deep-smoke-machine/back-end/data/extracted-data/frissewind'
    # Check
    if len(argv) > 1:
        if argv[1] != "confirm":
            print("Must confirm by running: python get_metadata.py confirm")
            return
    else:
        print("Must confirm by running: python get_metadata.py confirm")
        return

    # Get metadata

    print("Get video metadata from %s" % vid_path)
    vm = get_video_metadata(vid_path)

    # Save and return dataset
    file_path = "/home/paperspace/git/deep-smoke-machine/back-end/data/metadata-fresh-air-aug-17.json"
    check_and_create_dir(file_path)
    vm.to_json(file_path)

    print("Done saving video metadata to: " + file_path)


# Get video metadata from the video-labeling-tool
def get_video_metadata(vid_path):
    neg_examples = os.listdir(os.path.join(vid_path, 'Neg'))
    pos_examples = os.listdir(os.path.join(vid_path, 'Pos'))
    examples = ['Neg/'+ x for x in neg_examples] + ['Pos/' + x for x in pos_examples]
    labels = [0]*len(neg_examples) + [1]*len(pos_examples)
    df = pd.DataFrame({'file_name': examples, 'label': labels})
    df['file_name'] = df['file_name'].str.replace('.mp4', '')
    return df


# # Query a paginated api call iteratively until getting all the data
# def iterative_query(url, user_token, page_size=1000, page_number=1):
#     r = requests.post(url=url, data={"user_token": user_token, "pageSize": page_size, "pageNumber": page_number})
#     r = r.json()
#     total = r["total"]
#     r = r["data"]
#     if page_size*page_number >= total:
#         return r
#     else:
#         return r + iterative_query(url, user_token, page_size=page_size, page_number=page_number+1)


if __name__ == "__main__":
    main(sys.argv)
