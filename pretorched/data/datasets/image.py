import os
from PIL import Image

import torch.utils.data as data


class ImageDataset(data.Dataset):

    extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.JPEG']

    def __init__(self, root, metafile, transform=None, target_transform=None):
        self.root = root
        self.metafile = metafile
        self.transform = transform
        self.target_transform = target_transform
        self.imgs, self.classes = self._make_dataset()

        if len(self.imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 files in subfolders of: " + root + "\n"
                    "Supported extensions are: " + ",".join(self.extensions)
                )
            )

    def _make_dataset(self):
        imgs = []
        classes = set()
        with open(self.metafile) as f:
            for line in f.readlines():
                filename, target = line.strip().split(' ')
                target = int(target)
                path = os.path.join(self.root, filename)
                if any(path.endswith(ext) for ext in self.extensions):
                    imgs.append((path, target))
                    classes.add(target)
        return imgs, classes

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self._pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def _pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '  Number of datapoints: {}\n'.format(len(self.imgs))
        fmt_str += '  Number of classes: {}\n'.format(len(self.classes))
        fmt_str += '  Image list: {}\n'.format(self.metafile)
        tmp = '  Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp))
        )
        return fmt_str


class ImageFolder(ImageDataset):
    def __init__(
        self, root, metafile, transform=None, target_transform=None, classes=None
    ):
        self.classes = classes
        super().__init__(
            root, metafile, transform=transform, target_transform=target_transform
        )

    def _make_dataset(self):
        imgs = []
        if self.classes is not None:
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        else:
            self.classes, self.class_to_idx = self._find_classes()
        self.num_classes = len(self.classes)
        with open(self.metafile) as f:
            for line in f.readlines():
                path = os.path.join(self.root, line.strip())
                if any(path.endswith(ext) for ext in self.extensions):
                    target = self.class_to_idx[path.split('/')[-2]]
                    imgs.append((path, target))
        return imgs, self.classes

    def _find_classes(self):
        classes = set()
        with open(self.metafile) as f:
            for line in f.readlines():
                # filename, target = line.strip().split(' ')
                classes.add(line.strip().split('/')[-2])
        classes = sorted(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        return classes, class_to_idx


class ImageDir(ImageDataset):

    extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.JPEG']

    def __init__(self, root, transform=None, target_transform=None, default_target=0):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.default_target = default_target
        self.imgs, self.classes = self._make_dataset()

        if len(self.imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 files in subfolders of: " + root + "\n"
                    "Supported extensions are: " + ",".join(self.extensions)
                )
            )

    def _make_dataset(self):
        imgs = []
        classes = set()
        for filename in os.listdir(self.root):
            target = int(self.default_target)
            path = os.path.join(self.root, filename)
            if any(path.endswith(ext) for ext in self.extensions):
                imgs.append((path, target))
                classes.add(target)
        return imgs, classes
