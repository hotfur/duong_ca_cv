"""
Program to mass extract videos frames
@author: Scott
@author: Vuong Kha Sieu
"""
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def video_to_frames(video, path_output_dir):
    output = Path(path_output_dir)
    if not output.exists():
        output.mkdir(parents=True)
    """Extract frames from a video and save to directory as 'x.png' where x is the frame index
    @author: Scott
    https://stackoverflow.com/users/4663466/scott"""
    vidcap = cv2.VideoCapture(video)
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()


if __name__ == "__main__":
    # root directory of repo for relative path specification.
    root = Path(__file__).parent.absolute()
    # Set path to the images
    videos_path = root.joinpath("../../../data/line_trace/bacho")
    videos = videos_path.glob('*.mp4')
    with ThreadPoolExecutor() as executor:
        for fname in videos:
            output = Path(root.parent.joinpath("data").joinpath(fname.name).__str__().split(".")[0])
            if not output.exists():
                output.mkdir(parents=True)
            executor.submit(video_to_frames, root.joinpath(fname).__str__(), output)

