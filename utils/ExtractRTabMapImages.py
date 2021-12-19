import argparse
import os

parser = argparse.ArgumentParser(description='Extract images from .bag files')
parser.add_argument("directory", help="Path to the Experiments directory")

args = parser.parse_args()


for root, dirs, files in os.walk(args.directory):
    for file in files:
        if file.endswith(".db"):
            video_name = file.split('.')[0]
            # print(f"{root}\n{video_name}")
            print("\n\n-----------------------------------------\n\n")
            print(root)
            print("\n\n-----------------------------------------\n\n")
            os.chdir(root)
            os.mkdir(f"Images/{video_name}")
            os.system(f"rtabmap-export --images --output_dir 'Images/{video_name}' '{video_name}.db'")