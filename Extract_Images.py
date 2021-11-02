import argparse
import os

parser = argparse.ArgumentParser(description='Extract images from .bag files')
parser.add_argument("directory", help="Path to the Experiments directory")

args = parser.parse_args()


for root, dirs, files in os.walk(args.directory):
    for file in files:
        if file.endswith(".bag"):
            index = root.rfind('/')
            video_name = root[root.rfind('/')+1:]
            #print(f"{root}\n{video_name}")
            print("\n\n-----------------------------------------\n\n")
            print(root)
            print("\n\n-----------------------------------------\n\n")
            os.chdir(root)
            os.mkdir("Frames")
            os.system("roslaunch export.launch")
            os.system("mv ~/.ros/frame*.jpg Frames")
            os.system(f"ffmpeg -framerate 15 -i Frames/frame%04d.jpg -c:v \
                libx264 -profile:v high -crf 20 -pix_fmt yuv420p {video_name}.mp4")
            os.system(f"zip -r {video_name}_Frames.zip 'Frames'")
            os.system(f"rtabmap-export --images {video_name}.db")
            os.system(f"zip -r {video_name}_RTabMap_Images.zip {video_name}_rgb {video_name}_depth camera.yaml")