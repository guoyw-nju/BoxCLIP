import json
import torchvision
import torchvision.transforms as transforms


def get_datasets(parameters):

    # load dataset
    root = '/userhome/37/ywguo/summer-intern/datasets/{}2017'
    annFile = '/userhome/37/ywguo/summer-intern/datasets/annotations/instances_{}2017_new.json'

    datasets = {}
    for n in ('train', 'val'):
        datasets[n] = torchvision.datasets.CocoDetection(root.format(n), annFile.format(n), transform=transforms.Compose([transforms.ToTensor()]))
        print(f'{n} set scale: {len(datasets[n])}')

    with open(annFile.format('val')) as f:
        data = json.load(f)
        categories = {c['id']: c['name'] for c in data['categories']}

    return datasets, categories