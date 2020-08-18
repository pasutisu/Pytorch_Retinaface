import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from PIL import Image

# converting youtube faces npz format to widerface format
def convert_npz_to_widerface(npz_path, dst_path):
    image_path_list = []

    npz = np.load(npz_path)
    npz_name = os.path.basename(npz_path).split('.')[0]


    # transpose for iterate easily
    images = npz['colorImages'].transpose(3, 0, 1, 2) # n x y c

    image_x  = images.shape[1]
    image_y  = images.shape[2]

    pillow_images = [
        Image.fromarray(image) for image in images
    ]

    # transpose for iterate easily
    bboxes = npz['boundingBox'].transpose(2, 0, 1) # n x y
    # transpose for iterate easily
    landms = npz['landmarks2D'].transpose(2, 0, 1) # n x y

    # export jpg and label
    os.makedirs(os.path.join(dst_path, 'images/'), exist_ok=True)
    for idx, image in enumerate(pillow_images):
        image_path = '{}_{}.jpg'.format(npz_name, idx)
        image_path_list.append(
            '# {}\n{} {} {} {} {:6f} {:6f} 1.0 {:6f} {:6f} 1.0 {:6f} {:6f} 1.0 {:6f} {:6f} 1.0 {:6f} {:6f} 0.5'.format(
                image_path,
                int(bboxes[idx][0][0]),
                int(bboxes[idx][0][1]),
                int(bboxes[idx][3][0] - bboxes[idx][0][0]),
                int(bboxes[idx][3][1] - bboxes[idx][0][1]),
                landms[idx][36][0],
                landms[idx][36][1],
                landms[idx][45][0],
                landms[idx][45][1],
                landms[idx][31][0],
                landms[idx][31][1],
                landms[idx][49][0],
                landms[idx][49][1],
                landms[idx][55][0],
                landms[idx][55][1]
            )
        )

        image.save(os.path.join(dst_path, 'images/', image_path), quality=95)

    return image_path_list



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=str, help='dataset dir path')
    parser.add_argument('label_file', type=str, help='label.txt file path')
    parser.add_argument('--max_workers', type=int, help='label.txt file path', default=8)
    return parser.parse_args(argv)


def main(dataset_dir: str, label_file: str, max_workers: int):
    image_path_list = []

    dst_dir = os.path.abspath(os.path.dirname(label_file))
    os.makedirs(dst_dir, exist_ok=True)

    # preprocess for pararel processing    
    args_list = [ os.path.abspath(os.path.join(dataset_dir + npz_name)) for npz_name in os.listdir(dataset_dir) ]
    binded_convert_dataset_to_widerface = partial(convert_npz_to_widerface, dst_path=dst_dir)

    # make CPU hotplate
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(binded_convert_dataset_to_widerface, args_list):
           image_path_list.extend(result) 
    # for args in args_list:
    #     binded_convert_dataset_to_widerface(args)

    # output label.txt
    with open(label_file, 'w') as image_path_list_file:
        image_path_list_file.write('\n'.join(image_path_list))


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args.dataset_dir, args.label_file, args.max_workers)