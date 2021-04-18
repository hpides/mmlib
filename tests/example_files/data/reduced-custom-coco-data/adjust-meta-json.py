import json
import os

COCO_META = 'coco_meta'

if __name__ == '__main__':
    location = '.'
    location = os.path.abspath(location)
    image_names = []

    # r=>root, d=>directories, f=>files
    for r, d, f in os.walk(location):
        for item in f:
            if '.jpg' in item:
                image_names.append(item)

    new_meta_list = []

    # filter json
    with open('original_coco_meta.json', 'r') as myfile:
        data = myfile.read()

    # parse file
    obj = json.loads(data)
    meta_list = obj[COCO_META]

    for m in meta_list:
        if m['file_name'] in image_names:
            new_meta_list.append(m)

    with open("coco_meta.json", "w") as outfile:
        json.dump({COCO_META: new_meta_list}, outfile)

    print('test')
