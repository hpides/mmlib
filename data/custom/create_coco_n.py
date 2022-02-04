import argparse
import json
import os
import shutil

from experiments.data.custom.custom_coco import CustomCoco, FILE_NAME, COCO_META_JSON, COCO_META

IMAGES = 'images'


def main(args):
    data = CustomCoco(root=args.data_root, id_subset_json=args.id_subset_json, num_samples=args.size)

    root_path = os.path.abspath(args.dst_path)
    images_path = os.path.join(root_path, IMAGES)

    os.mkdir(root_path)
    os.mkdir(images_path)

    # copy all images
    included_file_names = []
    for sample in data._items:
        src_path = os.path.join(args.data_root, IMAGES, sample[FILE_NAME])
        dst_path = os.path.join(root_path, IMAGES, sample[FILE_NAME])
        shutil.copy(src_path, dst_path)
        included_file_names.append(sample[FILE_NAME])

    # create reduced json
    current_json_path = os.path.join(args.data_root, COCO_META_JSON)
    new_coco_meta_list = []
    with open(current_json_path) as f:
        j_doc = json.load(f)
        coco_meta = j_doc[COCO_META]
        for e in coco_meta:
            file_name = e[FILE_NAME]
            if file_name in included_file_names:
                new_coco_meta_list.append(e)

    j_doc[COCO_META] = new_coco_meta_list

    with open(os.path.join(root_path, COCO_META_JSON), 'w') as f:
        json.dump(j_doc, f)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--size', help='the size in number of samples of the dataset', type=int, required=True)
    parser.add_argument('--data-root', help='the root path to teh custom coco dataset', type=str, required=True)
    parser.add_argument('--id-subset-json', help='file to specify which ids are included', type=str, required=True)
    parser.add_argument('--dst-path', help='the path where the new dataset is created', type=str, required=True)

    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    args = parse_args()
    main(args)
