import argparse
import cv2 as cv
from dataset import dataset
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ucf', type=str, help='Dataset: [ucf, hmdb, olympics, test]')
parser.add_argument('--dataset_path', required=True, type=str, help='Path of the datasets')

parser.add_argument('--clip_len', default=16, type=int, help='Number of frames of each sample clip')
parser.add_argument('--n_clips', default=1, type=int, help='Number of clips per video (training)')
parser.add_argument('--n_videos', default=2, type=int, help='Number of videos to sample for each class')
parser.add_argument('--class_name', default='all', type=str, help='Class name, \'all\' for all classes')

parser.add_argument('--save_path', required=True, type=str, help='Where to save log and checkpoint.')
opt = parser.parse_args()

if __name__ == '__main__':
    ucf_fnames, ucf_labels, ucf_classes, folder = dataset.get_test_data(opt.dataset, opt.dataset_path)
    if opt.class_name not in ucf_classes and opt.class_name != 'all':
        print("No such class name")
        exit()

    if os.path.exists(opt.save_path) == False:
        os.mkdir(opt.save_path)
        os.mkdir(os.path.join(opt.save_path, 'frame'))

    dataset_path = os.path.join(opt.save_path, 'frame', opt.dataset)
    if os.path.exists(dataset_path) == False:
        os.mkdir(dataset_path)

    for ucf_class in ucf_classes:
        if ucf_class == opt.class_name or opt.class_name == 'all':
            os.mkdir(os.path.join(dataset_path, ucf_class))

    mp = {}
    for fname in ucf_fnames:
        new_fname = os.path.relpath(fname, start=folder)
        new_fname = new_fname[:-4]
        class_name, _ = new_fname.split('/')
        # print('Fetching %d frames of %s', opt.clip_len, class_name)
        if class_name == opt.class_name or opt.class_name == 'all':
            if class_name not in mp:
                mp[class_name] = 1
            elif mp[class_name] >= opt.n_videos:
                continue
            else:
                mp[class_name] += 1

            frames = dataset.load_clips(fname, opt.clip_len, opt.n_clips, is_validation=False)
            for n_clip in range(opt.n_clips):
                for clip_len in range(opt.clip_len):
                    result = frames[n_clip][clip_len]
                    path = os.path.join(opt.save_path, 'frame', opt.dataset, new_fname + '_' + str(n_clip) + '_' + str(clip_len) + '.png')
                    cv.imwrite(path, result)