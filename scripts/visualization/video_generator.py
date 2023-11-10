import subprocess

output = subprocess.check_output("ls -hatr ./plots/*.png", shell=True)
frames = output.decode().split("\n")[:-1]

import cv2

frame = cv2.imread(frames[0])
height, width, layers = frame.shape
video = cv2.VideoWriter('./plots/video.avi', 0, 1, (frame.shape[1], frame.shape[0]))

for j in frames:
    img = cv2.imread(j)
    video.write(img)

cv2.destroyAllWindows()
video.release()
