import os
import sys
from util import *
import numpy as np
from optical_flow.optical_flow import OpticalFlow
from multiprocessing import Pool
import pandas as pd

thread = "1"
os.environ["MKL_NUM_THREADS"] = thread
os.environ["NUMEXPR_NUM_THREADS"] = thread
os.environ["OMP_NUM_THREADS"] = thread
os.environ["VECLIB_MAXIMUM_THREADS"] = thread
os.environ["OPENBLAS_NUM_THREADS"] = thread
import cv2 as cv
cv.setNumThreads(12)
from tqdm import tqdm

rgb_dir = "/mnt/sdb/data/frissewind-npy-2/rgb"
flow_dir = "/mnt/sdb/data/frissewind-npy-2/flow"
video_dir = "/mnt/sdb/data/frissewind-videos"

# Process videos into rgb frame files and optical flow files
# The file format is numpy.array
def main(argv):    
    metadata_path = "/home/pravar_d_mahajan/git/deep-smoke-machine/back-end/data/metadata-frissewind.json"
    num_workers = 4

    # Check for saving directories and create if they don't exist
    check_and_create_dir(rgb_dir)
    check_and_create_dir(flow_dir)

    metadata = pd.read_json(metadata_path)
    #print("Starting to create pool.")
    #p = Pool(num_workers)
    tqdm.pandas()
    metadata.progress_apply(compute_and_save_flow, axis=1)
    print("Done process_videos.py")


def compute_and_save_flow(video_data):
    file_name = video_data["file_name"]
    rgb_vid_in_p = os.path.join(video_dir, file_name + ".mp4")
    if not os.path.exists(rgb_vid_in_p): return
    rgb_4d_out_p = os.path.join(rgb_dir, file_name + ".npy")
    flow_4d_out_p = os.path.join(flow_dir, file_name + ".npy")
    #print("Processing: {}".format(file_name, rgb_4d_out_p, flow_4d_out_p))
    # Saves files to disk in format (time, height, width, channel) as numpy array
    flow_type = 1 # TVL1 optical flow
    #flow_type = None # will not process optical flow
    op = OpticalFlow(rgb_vid_in_p=rgb_vid_in_p, rgb_4d_out_p=rgb_4d_out_p,
            flow_4d_out_p=flow_4d_out_p, clip_flow_bound=20, flow_type=flow_type)
    op.process()


if __name__ == "__main__":
    main(sys.argv)
