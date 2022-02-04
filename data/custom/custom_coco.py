import json
import os
from typing import Optional, Callable, Any

from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset

FILE_NAME = 'file_name'
IMAGES = 'images'
INCLUDED_COCO_IDS = 'included-coco-ids'
COCO_META = 'coco_meta'
COCO_META_JSON = 'coco_meta.json'
COCO_IMAGE_TYPE = ".jpg"
COCO_FILENAME_NUMBERS = 12
COCO_CLASSES = 91
COCO_SPLIT = 'coco_split'
COCO_CATEGORY_ID = 'coco_category_id'
COCO_IMAGE_ID = 'coco_image_id'
IMAGENET_WNID = 'imagenet_wnid'
IMAGENET_CLASS_ID = 'imagenet_class_id'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

inference_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize, ])

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])


def _included_ids(id_subset_json):
    with open(id_subset_json, 'r') as json_file:
        included_id_json = json.load(json_file)
        ids = included_id_json[INCLUDED_COCO_IDS]
    return ids


class CustomCoco(VisionDataset):

    def __init__(self,
                 root: str,
                 ann_file: str = COCO_META_JSON,
                 id_subset_json: str = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,
                 num_samples: int = None
                 ) -> None:
        super(CustomCoco, self).__init__(root, transforms, transform, target_transform)
        self.images_path = os.path.join(self.root, IMAGES)
        self.ann_file = os.path.join(self.root, ann_file)

        if id_subset_json:
            self.included_ids = _included_ids(id_subset_json)
        else:
            self.included_ids = list(range(COCO_CLASSES + 1))

        self._items = []
        with open(self.ann_file) as f:
            ann_data = json.load(f)
            coco_meta = ann_data[COCO_META]
            for e in coco_meta:
                coco_cat = e[COCO_CATEGORY_ID]
                if coco_cat in self.included_ids:
                    self._items.append(e)

        if num_samples is not None:
            if len(self._items) >= num_samples:
                self._items.sort(key=lambda x: x[COCO_IMAGE_ID])
                self._items = self._items[:num_samples]
                assert len(self._items) == num_samples
            else:
                raise ValueError('The given num_samples is higher than the available number of samples')

    def __getitem__(self, index: int) -> Any:
        item = self._items[index]
        image_path = os.path.join(self.images_path, item[FILE_NAME])
        img = Image.open(image_path).convert('RGB')
        label = item[IMAGENET_CLASS_ID]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def __len__(self) -> int:
        return len(self._items)


class InferenceCustomCoco(CustomCoco):

    def __init__(self, root: str, ann_file: str = COCO_META_JSON, id_subset_json: str = None,
                 target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None,
                 num_samples: int = None):
        transform = inference_transforms
        super().__init__(root, ann_file, id_subset_json, transform, target_transform, transforms, num_samples)


class TrainCustomCoco(CustomCoco):

    def __init__(self, root: str, ann_file: str = COCO_META_JSON, id_subset_json: str = None,
                 target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None,
                 num_samples: int = None):
        transform = train_transforms
        super().__init__(root, ann_file, id_subset_json, transform, target_transform, transforms, num_samples)
