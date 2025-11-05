import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import glob

class CenterCropLongEdge(object):
    """Crops the given PIL Image on the long edge.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageDataset(data.Dataset):

    def __init__(self,
                 root_dir,
                 meta_file=None,
                 transform=None,
                 image_size=128,
                 normalize=True,
                 max_images=None,
                 image_extensions=None):
        """
        Args:
            root_dir: Directory containing images
            meta_file: Optional text file with image names and labels. If None, automatically scans directory.
            transform: Optional transform
            image_size: Target image size
            normalize: Whether to normalize images
            max_images: Maximum number of images to load (None = all)
            image_extensions: List of image extensions to look for (default: ['jpg', 'jpeg', 'png', 'JPEG', 'JPG', 'PNG'])
        """
        self.root_dir = root_dir
        if transform is not None:
            self.transform = transform
        else:
            norm_mean = [0.5, 0.5, 0.5]
            norm_std = [0.5, 0.5, 0.5]
            if normalize:
                self.transform = transforms.Compose([
                    CenterCropLongEdge(),
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std)
                ])
            else:
                self.transform = transforms.Compose([
                    CenterCropLongEdge(),
                    transforms.Resize(image_size),
                    transforms.ToTensor()
                ])
        
        self.metas = []
        self.classifier = None
        
        if meta_file is not None and os.path.exists(meta_file):
            # Load from meta file (original behavior)
            with open(meta_file) as f:
                lines = f.readlines()
            print("building dataset from %s" % meta_file)
            self.num = len(lines)
            # suffix = ".jpeg"
            suffix = ""
            for line in lines:
                line_split = line.rstrip().split()
                if len(line_split) == 2:
                    self.metas.append((line_split[0] + suffix, int(line_split[1])))
                else:
                    self.metas.append((line_split[0] + suffix, -1))
            print("read meta done")
        else:
            # Automatically scan directory for images
            if image_extensions is None:
                image_extensions = ['jpg', 'jpeg', 'png', 'JPEG', 'JPG', 'PNG']
            
            print("automatically scanning directory: %s" % root_dir)
            image_files = []
            for ext in image_extensions:
                # Search in root_dir and subdirectories
                pattern = os.path.join(root_dir, '**', '*.' + ext)
                image_files.extend(glob.glob(pattern, recursive=True))
                # Also search directly in root_dir
                pattern = os.path.join(root_dir, '*.' + ext)
                image_files.extend(glob.glob(pattern, recursive=False))
            
            # Remove duplicates and sort
            image_files = sorted(list(set(image_files)))
            
            # Limit number of images if specified
            if max_images is not None:
                image_files = image_files[:max_images]
            
            self.num = len(image_files)
            print("found %d images" % self.num)
            
            # Store relative paths from root_dir
            for img_path in image_files:
                rel_path = os.path.relpath(img_path, root_dir)
                # Normalize path separators
                rel_path = rel_path.replace('\\', '/')
                self.metas.append((rel_path, -1))  # No class label available
            
            print("scanning done")

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        rel_path = self.metas[idx][0]
        # Use os.path.join for proper path handling
        filename = os.path.join(self.root_dir, rel_path)
        cls = self.metas[idx][1]
        img = default_loader(filename)

        # transform
        if self.transform is not None:
            img = self.transform(img)

        return img, cls #, self.metas[idx][0]