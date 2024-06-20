import cv2
import subprocess
import sys
from datetime import datetime

output = subprocess.check_output(f"ls -hatr {sys.argv[1]}/*.{sys.argv[2]}", shell=True)
frames = output.decode().split("\n")[:-1]

W = None
try:
    W = int(sys.argv[3])
except ValueError:
    W = frames.shape[1]

H = None
try:
    H = int(sys.argv[4])
except ValueError:
    H = frames.shape[0]

frame = cv2.imread(frames[0])
height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for the output video
video = cv2.VideoWriter(f'{sys.argv[1]}/video_{datetime.now().timestamp()}.mp4', fourcc, 45, (W, H))

for j in frames:
    img = cv2.imread(j)
    video.write(img)

cv2.destroyAllWindows()
video.release()


# import cv2
# import os
#
# image_dir = "/home/leandro/PhD/data/KITTI/visualization/video_sequence_d4lcn_100_occlusion_rotated_box_with_lines/"
# output_file = "/home/leandro/PhD/data/KITTI/visualization/video_sequence_d4lcn_100_occlusion_rotated_box_with_lines.mp4"
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for the output video
#
# image_dir_0 = "/home/leandro/PhD/data/KITTI/visualization/video_sequence_d4lcn/sequence_1/"
# image_files = os.listdir(image_dir_0)
# image_files.sort(key=lambda x: int(x.split('.')[0]))
#
# img0 = cv2.imread(os.path.join(image_dir_0, image_files[0]))
# height, width, channels = img0.shape
# frame_rate = 10
# out = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))
#
# sequence_files = os.listdir(image_dir)
# for sequence in sequence_files:
#     image_files = os.listdir(os.path.join(image_dir, sequence))
#     image_files.sort(key=lambda x: int(x.split('.')[0]))
#     for image_file in image_files:
#         img = cv2.imread(os.path.join(image_dir, sequence, image_file))
#         out.write(img)
#
# out.release()