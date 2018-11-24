# other.py

from typing import Union, Any, Callable, Tuple, Dict, Iterable, List
from pathlib import Path
import pickle
import time
from functools import wraps
from urllib.request import urlopen
import sys
from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor

import pandas as pd
import json
import numpy as np
import PIL
from IPython.display import display


def new_save(out_path: Path, data, file_format: str='pickle'):
    """(Over)write data to new pickle/json file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if file_format == 'pickle':
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
    elif file_format == 'json':
        with open(out_path, "w") as f:
            json.dump(data, f, indent=4)


def read_saved(in_path: Path, file_format: str='pickle'):
    """Read saved pickle/json file."""
    if file_format == 'pickle':
        with open(in_path, "rb") as f:
            data = pickle.load(f)
    elif file_format == 'json':
        with open(in_path, "r") as f:
            data = json.load(f)
    return data


def read_or_new_save(path: Path,
                     default_data: Union[Callable, Any],
                     callable_args: Dict=None,
                     file_format: str='pickle'
                     ) -> Any:
    """Write data to new pickle/json file or read pickle/json if that file already exists.

    Example:
        df = cgeo.other.read_or_new_save(path=Path('output\preprocessed_marker_small.pkl'),
                                         default_data=preprocess_vector,
                                         callable_args={'inpath': fp_fields, 'meta': meta})
    Args:
        path: in/output pickle/json file path.
        file_format: Either 'pickle' or 'json'.
        default_data: Data that is written to a pickle/json file if the pickle/json does not already exist.
            When giving a function, do not call the function, only give the function
            object name. Function arguments can be provided via callable_args.
        callable_args: args for additional function arguments when default_data is a callable function.

    Returns:
        Contents of the read or newly created pickle/json file.
    """
    try:
        if file_format == 'pickle':
            data = read_saved(path, file_format=file_format)
        elif file_format == 'json':
            data = read_saved(path, file_format=file_format)
        print(f'Reading from {file_format} file... {path.name}')
    except (FileNotFoundError, OSError, IOError, EOFError):
        if not callable(default_data):
            data = default_data
        else:
            if callable_args is None:
                data = default_data()
            else:
                data = default_data(**callable_args)
        print(f'Writing new {file_format} file... {path.name}')
        if file_format == 'pickle':
            new_save(out_path=path, data=data, file_format=file_format)
        elif file_format == 'json':
            new_save(out_path=path, data=data, file_format=file_format)

    return data


def lprun(func):
    """Line profile decorator.

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
    """Displays full dataframe (deactivates rows/columns wrapper). Prints if not in Notebook."""
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(df)


def sizeof_memvariables(locals):
    """Prints size of all variables in memory in human readable output.
    
    By Fred Cirera, after https://stackoverflow.com/a/1094933/1870254
    """

    def sizeof_fmt(num, suffix='B'):
        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, 'Yi', suffix)

    for name, size in sorted(((name, sys.getsizeof(value)) for name,value in locals().items()),
                             key=lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
    

def download_url(url, out_path):
    """Download file from URL.

    Example: download_file("url", 'data.tif')
    """
    print('downloading {} to {}'.format(url, out_path))
    with open(out_path, "wb") as local_file:
        local_file.write(urlopen(url).read())


def print_file_tree(dir: Path=None):
    """Print file tree of the selected directory.

    Taken from https://realpython.com/python-pathlib/

    Args:
        dir: The directory to print the file tree for. Defaults to current working directory.
    """
    if dir is None:
        dir = Path.cwd()
    print(f'+ {dir}')
    for path in sorted(dir.rglob('*')):
        depth = len(path.relative_to(dir).parts)
        spacer = '    ' * depth
        print(f'{spacer}+ {path.name}')


def track_time(task):
    """Track time start/end of running function."""
    start_time = time.time()
    state = task.status()['state']
    print('RUNNING...')
    while state in ['READY', 'RUNNING']:
        time.sleep(3)
        state = task.status()['state']
    elapsed_time = time.time() - start_time
    print('Done in', elapsed_time, 's')
    print(task.status())


def multithread_iterable(func: Callable,
                         iterable: Iterable,
                         func_kwargs: Dict=None,
                         max_workers: int=2,
                         iter_is_tuple=False):
    """Wrapper for simplified multithreading of iterable.

    Uses concurrent.futures.ThreadPoolExecutor instead of manually spinning up threads via the threading module.

    Args:
        func: callable function.
        iterable: list, generator etc. that should be iterated over via one thread per iteration. If the iterable
            yields a tuple,
        func_kwargs: additional function arguments.
        max_workers: number of threads.
        iter_is_tuple: Set True if iterable yields tuples.

    Returns:
        The function return value in a list.

    Example:
        def task(i, iter, add=2):   # i and iter are required arguments!
            print("Processing {}".format(i))
            return iter*iter + add
        print(multithreading(func=task, iterable=[2,3,4], func_kwargs={'add':10}, max_workers=2))
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if not iter_is_tuple:
            futures = [executor.submit(func, i, iter, **func_kwargs) for i, iter in enumerate(iterable)]
        else:
            futures = [executor.submit(func, i, *iter, **func_kwargs) for i, iter in enumerate(iterable)]

        res = [fut.result() for fut in as_completed(futures)]
        return res


def roman_numbers_to_arrays(text_list: List[str],
                            fontsize: int = 12,
                            display=True
                            ) -> List[np.array]:
    """Create binary arrays displaying Roman numbers.

    Inspired by https://stackoverflow.com/questions/36384353/generate-pixel-matrices-from-characters-in-string
    Args:
        text_list: List of Roman numbers as string. Defaults to I-X.
        fontsize: Should be at least 12, otherwise deformations
        display: In addition to returning the arrays plot them.

    Returns:
        List of binary numpy arrays, all with the same dimensions.

    Example: roman_arrays = roman_to_pixels(['I', 'II'], 22, display=False)
    """
    font = PIL.ImageFont.truetype('arialbd.ttf', fontsize)

    if not text_list:
        roman = {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI', 7: 'VII', 8: 'VIII', 9: 'IX', 10: 'X'}
        text_list = list(roman.values())
    if fontsize < 12:
        raise ValueError('fontsize needs to be at least 12, smaller will cause font deformations')

    widths = []
    for text in text_list:
        w, h = font.getsize(text)  # calc the size of text in pixels
        h *= 2
        widths.append(w)
    w, h = max(widths), h

    arrays = []
    for text in text_list:
        image = PIL.Image.new('L', (w, h), 1)
        draw = PIL.ImageDraw.Draw(image)
        draw.text((0, 0), text, font=font)
        arr = np.asarray(image)
        arr = np.where(arr, 0, 1)
        arr = arr[(arr != 0).any(axis=1)]
        arrays.append(arr)

        if display is True:
            result = np.where(arr, '#', ' ')
            print('shape', arr.shape)
            print('\n'.join([''.join(row) for row in result]))

    return arrays
