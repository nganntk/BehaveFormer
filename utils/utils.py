import os
import pickle

def read_pickle(pickle_path: str) -> object:
    """Read pickle file and return the ata

    Args:
        pickle_path (str): Absolute path to the pickle file

    Returns:
        object: Data from the pickle file
    """    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data


def list2txt(data_list: list, out_file: str, allow_overwrite: bool=False):
    """Write a list to a txt file, each item in the list is a line in the txt file.

    Args:
        data_list (list): List of items to write to the txt file.
        out_file (str): Path to the output txt file.
        allow_overwrite (bool): Whether to allow overwriting the output file if it exists.
    """
    if not allow_overwrite and os.path.exists(out_file):
        raise FileExistsError(f'{out_file} exists!')

    with open(out_file, 'w') as f:
        for item in data_list:
            f.write('{}\n'.format(item))
