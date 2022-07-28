import json
from src.datasets.coco import CocoDetection
import torchvision
import torchvision.transforms as transforms


def get_datasets(parameters, clip_preprocess):

    # load dataset
    root = '/userhome/37/ywguo/summer-intern/datasets/{}2017'
    annFile = '/userhome/37/ywguo/summer-intern/datasets/annotations/instances_{}2017_new.json'
    annFile_cap = '/userhome/37/ywguo/summer-intern/datasets/annotations/captions_{}2017.json'

    datasets = {}
    for n in ('train', 'val'): 
        datasets[n] = CocoDetection(root=root.format(n), annFile=annFile.format(n), annFile_cap=annFile_cap.format(n), transform=transforms.Compose([transforms.Resize([224, 224]), clip_preprocess]))

        print(f'{n} set scale: {len(datasets[n])}')

    return datasets