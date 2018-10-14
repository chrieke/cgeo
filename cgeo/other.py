# other.py

from typing import Union, Any, Callable
from pathlib import Path
import pickle
import time
from functools import wraps
from urllib.request import urlopen

import pandas as pd
from IPython.display import display


def new_pickle(out_path: Path, data):
    """
    Write data to new pickle file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(data, f)


def read_or_new_pickle(path: Path, default_data: Union[Callable, Any]) -> Any:
    """
    Write data to new pickle file or read pickle if that file already exists.

    Args:
        path (Path): in/output pickle file path.
        default_data (Union[Callable, Any]): Data that is written to a pickle file if the pickle does not already exist.
                                             When giving a function, do not call the function, only give the function
                                             object name. Does currently not accept additional function arguments.

    Returns (Any): Contents of the read or newly created pickle file.
    """
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
            print('Reading from pickle file...')
    except (FileNotFoundError, OSError, IOError, EOFError):
        if callable(default_data):
            data = default_data()
        else:
            data = default_data
        print('Writing new pickle file...')
        new_pickle(out_path=path, data=data)
    return data


def lprun(func):
    """
    Line profile decorator.

    Put @lprun on the function you want to profile.
    From pavelpatrin: https://gist.github.com/pavelpatrin/5a28311061bf7ac55cdd
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        from line_profiler import LineProfiler
        prof = LineProfiler()
        try:
            return prof(func)(*args, **kwargs)
        finally:
            prof.print_stats()
    return wrapper


def printfull(df):
    """
    Displays full dataframe. Prints if not in Notebook.
    """
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(df)


def download_url(url, out_path):
    """
    Download file from URL.

    Example: download_file("url", 'data.tif')
    """
    print('downloading {} to {}'.format(url, out_path))
    with open(out_path, "wb") as local_file:
        local_file.write(urlopen(url).read())


def print_file_tree(dir: Path=None):
    """
    Print file tree.

    Taken from https://realpython.com/python-pathlib/

    Args:
        dir(Path): The directory to print the file tree for. Defaults to current working directory.
    """
    if dir is None:
        dir = Path.cwd()
    print(f'+ {dir}')
    for path in sorted(dir.rglob('*')):
        depth = len(path.relative_to(dir).parts)
        spacer = '    ' * depth
        print(f'{spacer}+ {path.name}')


def track_time(task):
    """
    Track time start/end of running function.
    """
    start_time = time.time()
    state = task.status()['state']
    print('RUNNING...')
    while state in ['READY', 'RUNNING']:
        time.sleep(3)
        state = task.status()['state']
    elapsed_time = time.time() - start_time
    print('Done in', elapsed_time, 's')
    print(task.status())