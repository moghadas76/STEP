import subprocess

output = subprocess.check_output("ls -hatr ./plots/*.jpg", shell=True)
frames = output.decode().split("\n")[:-1]

import cv2

frame = cv2.imread(frames[0])
height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for the output video
video = cv2.VideoWriter('./plots/video.mp4', fourcc, 1, (frame.shape[1], frame.shape[0]))

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