from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
from typing import Any, Callable, Optional, Tuple, List

import json

class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        annFile_cap: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.coco_caption = COCO(annFile_cap)
        self.ids = list(sorted(self.coco.imgs.keys()))

        with open(annFile) as f:
            js = json.load(f)
            self.cats = {c['id']: c['name'] for c in js['categories']}

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id) -> List[Any]:
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        # embed category'name into the datasets
        for ann in anns:
            ann['category_name'] = self.cats[ann['category_id']]
        return anns
        # return self.coco.loadAnns(self.coco.getAnnIds(id))

    def _load_caption(self, id) -> List[Any]:
        return [ann["caption"] for ann in self.coco_caption.loadAnns(self.coco_caption.getAnnIds(id))]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        w, h = image.size

        target = self._load_target(id)
        # normalize bbox's coordinate while maintaining the original data
        _target = [obj.copy() for obj in target]
        for i in range(len(_target)):
            _target[i]['bbox'] = [_target[i]['bbox'][0]/w, _target[i]['bbox'][1]/h, 
                                  _target[i]['bbox'][2]/w, _target[i]['bbox'][3]/h]

        caption = self._load_caption(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target) # [Resize, clip_process]

        return image, _target, caption

    def __len__(self) -> int:
        return len(self.ids)


class CocoCaptions(CocoDetection):
    """`MS Coco Captions <https://cocodataset.org/#captions-2015>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Example:

        .. code:: python

            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root = 'dir where images are',
                                    annFile = 'json annotation file',
                                    transform=transforms.ToTensor())

            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample

            print("Image Size: ", img.size())
            print(target)

        Output: ::

            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']

    """

    def _load_target(self, id) -> List[str]:
        return [ann["caption"] for ann in super()._load_target(id)]
