from cam_distort import undistort_file, unpack_yaml
import os
input_path = './cam3_images'
yaml_file = 'cam3.yaml'
output_path = './undistorted'
mapx, mapy = unpack_yaml(yaml_file)
for f in os.listdir(input_path):
        file = os.path.join(input_path, f)
        if os.path.isfile(file):
            name = f.split('.')
            undistort_file(file, mapx, mapy, output_path)