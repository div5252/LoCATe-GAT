import os
import numpy as np

def convert_olympics():
    folder = '../../datasets/olympic_sports/'
    output_folder = '../../datasets/olympic_sports_video/'

    for label in sorted(os.listdir(str(folder))):
        dir = os.path.join(str(folder), label)
        for fname in sorted(os.listdir(dir)):
            command = "ffmpeg -i "
            fname = os.path.join(str(folder), label, fname)
            command += fname + " "
            command += output_folder + fname[len(folder):-3] + "mp4"
            os.system(command)

if __name__ == '__main__':
    convert_olympics()
    