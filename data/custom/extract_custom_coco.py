import argparse
import json
import os
from shutil import copyfile

import torchvision
from termcolor import colored

import experiments.data.util.constants as const

OK = colored('OK', 'green')


def main(args):
    # create target root dir
    os.makedirs(args.target_root)

    print(OK, 'Read COCO train ...')
    train_coco_data = torchvision.datasets.CocoDetection(args.coco_train_root_path, args.coco_train_annotations)
    print(OK, 'done')

    print(OK, 'Read COCO val ...')
    val_coco_data = torchvision.datasets.CocoDetection(args.coco_val_root_path, args.coco_val_annotations)
    print(OK, 'done')

    # filter the data s. th. all samples only have __exactly__ one assigned category
    print(OK, 'Filter train ...')
    train_one_cat = filter_number_of_categories(train_coco_data, 1)
    print(OK, 'done')
    print(OK, 'found {} elements with exactly one category in COCO training data'.format(len(train_one_cat)))

    print(OK, 'Filter val ...')
    val_one_cat = filter_number_of_categories(val_coco_data, 1)
    print(OK, 'done')
    print(OK, 'found {} elements with exactly one category in COCO validation data'.format(len(val_one_cat)))

    print(OK, 'Load Imagenet data ...')
    imagenet_data = torchvision.datasets.ImageNet(args.imagenet_root, split=args.imagenet_split)
    print(OK, 'done')

    # both indexes for train and val will be the same
    coco_cat_index = id_to_class_index(args.coco_val_annotations)

    print(OK, 'Match data ...')
    val_match = match_classes(val_one_cat, coco_cat_index, imagenet_data.class_to_idx)
    train_match = match_classes(train_one_cat, coco_cat_index, imagenet_data.class_to_idx)
    print(OK, 'done')

    print(OK, 'Save data ...')
    val_match = [(x, const.VAL) for x in val_match]
    train_match = [(x, const.TRAIN) for x in train_match]
    matched = val_match + train_match
    save_matched_coco(matched, imagenet_data.wnids, args)
    print(OK, 'done')


def save_matched_coco(elements, wnids, args):
    meta_list = []

    for element, split in elements:
        coco_element, imagenet_index = element
        _, coco_meta = coco_element

        # we get a list but only need the first element
        # in case there are multiple elements they have the same category and the info is redundant
        coco_meta = coco_meta[0]

        meta_dict = create_meta_dict(coco_meta, imagenet_index, wnids, split)
        image_path = create_image_path(args, meta_dict[const.FILE_NAME], split)

        # add img metadata to list and cpy image to target dir
        meta_list.append(meta_dict)
        copy_image(image_path, args.target_root)

    # save the metadata to a json
    json_file_path = os.path.join(args.target_root, const.COCO_META_JSON)
    sava_coco_meta(json_file_path, meta_list)


def create_coco_filename(image_id):
    return str(image_id).zfill(const.COCO_FILENAME_NUMBERS) + const.COCO_IMAGE_TYPE


def copy_image(image_path, target_root):
    target_dir = os.path.join(target_root, const.IMAGES)

    # create dir if not exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    dst_path = os.path.join(target_dir, os.path.basename(image_path))
    copyfile(image_path, dst_path)


def sava_coco_meta(json_file_path, meta_list):
    coco_meta = {const.COCO_META: meta_list}
    with open(json_file_path, 'w') as json_file:
        json.dump(coco_meta, json_file)


def create_image_path(args, filename, split):
    if split == const.TRAIN:
        base_path = args.coco_train_root_path
    elif split == const.VAL:
        base_path = args.coco_val_root_path
    else:
        assert False

    return os.path.join(base_path, filename)


def create_meta_dict(coco_meta, imagenet_index, wnids, split):
    result_meta = {}
    # copy data from COCO data
    result_meta[const.COCO_IMAGE_ID] = coco_meta[const.IMAGE_ID]
    result_meta[const.COCO_CATEGORY_ID] = coco_meta[const.CATEGORY_ID]
    result_meta[const.COCO_SPLIT] = split

    result_meta[const.IMAGENET_CLASS_ID] = imagenet_index
    result_meta[const.IMAGENET_WNID] = wnids[imagenet_index]
    result_meta[const.FILE_NAME] = create_coco_filename(coco_meta[const.IMAGE_ID])

    return result_meta


def id_to_class_index(annotations_path):
    index = {}
    with open(annotations_path, 'r') as COCO:
        js = json.loads(COCO.read())
        cats = js[const.CATEGORIES]

        for cat in cats:
            index[cat[const.ID]] = cat[const.NAME]

    return index


def match_classes(val_one_cat, coco_index, imagenet_index):
    matched = []
    for element in val_one_cat:
        coco_category = get_coco_category(element, coco_index)
        if coco_category and coco_category in imagenet_index.keys():
            matched.append((element, imagenet_index[coco_category]))

    return matched


def get_coco_category(element, cat_index):
    _, annot = element
    cat_ids = category_ids(annot)
    # sometimes no category is set
    assert len(cat_ids) <= 1
    if len(cat_ids) == 1:
        cat = cat_index[cat_ids.pop()]
        return cat


def filter_number_of_categories(elements, num_categories):
    return [e for e in elements if len(category_ids(e[1])) == num_categories]


def category_ids(annotation):
    cat_ids = set()
    for a in annotation:
        cat_ids.add(a[const.CATEGORY_ID])

    return cat_ids


def parse_args():
    parser = argparse.ArgumentParser(description='Creation of customized COCO dataset')
    parser.add_argument('--coco-train-root-path', help='coco root path for training data; \'train2017\'', )
    parser.add_argument('--coco-train-annotations',
                        help='coco path for training annotations; \'instances_train2017.json\'')
    parser.add_argument('--coco-val-root-path', help='coco root path for validation data; \'val2017\'')
    parser.add_argument('--coco-val-annotations',
                        help='coco path for validation annotations; \'instances_val2017.json\'')

    parser.add_argument('--imagenet-root', help='imagenet root path for')
    parser.add_argument('--imagenet-split', default=const.VAL, choices=[const.TRAIN, const.VAL])

    parser.add_argument('--target-root', help='the directory to store the created dataset')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
