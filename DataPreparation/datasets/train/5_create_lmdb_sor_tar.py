from basicsr.utils import scandir
from basicsr.utils.lmdb import make_lmdb_from_imgs


def create_lmdb_for_ovarian():
    """Create lmdb files for data_ovarian dataset.

    Usage:
        Before run this script, please run `extract_subimages_sor_tar.py`.
        Typically, there are two folders to be processed for data_ovarian dataset.
            train_GT_rename_crop_sub
            train_input_rename_warped_crop_sub
        Remember to modify opt configurations according to your settings.
    """

    # HR images
    folder_path = 'datasets/data_ovarian/train_GT_rename_crop_sub'
    lmdb_path = 'datasets/data_ovarian/train_GT_rename_crop_sub.lmdb'
    img_path_list, keys = prepare_keys_ovarian(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # LRx4 images
    folder_path = 'datasets/data_ovarian/train_input_rename_warped_crop_sub'
    lmdb_path = 'datasets/data_ovarian/train_input_rename_warped_crop_sub.lmdb'
    img_path_list, keys = prepare_keys_ovarian(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


def prepare_keys_ovarian(folder_path):
    """Prepare image path list and keys for data_ovarian dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(
        list(scandir(folder_path, suffix='tif', recursive=False)))
    keys = [img_path.split('.tif')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys


if __name__ == '__main__':
    create_lmdb_for_ovarian()
