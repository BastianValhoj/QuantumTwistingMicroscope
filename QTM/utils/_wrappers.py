from time import time

# from utils import in_notebook

from tqdm.auto import tqdm

def timeit(func):
    """Decorator to time function call."""
    def wrapper(*args, **kwargs):
        initial_time = time()
        result = func(*args, **kwargs)
        final_time = time()
        # func_args = {}
        arg_str = ", ".join([
            *(repr(a) for a in args),
            *(f"{k}={v!r}" for k, v in kwargs.items())
        ]) or "-"
        time_diff = abs(final_time - initial_time)
        hours, rem = divmod(time_diff, 60*60)
        minutes, seconds = divmod(rem, 60)
        hours, minutes = int(hours), int(minutes)
        print(f"func '{func.__name__}'({arg_str})  ran in {hours:02d}:{minutes:02d}:{seconds:05.3f}")
        return result
    return wrapper

def count_removals(func):
    """Decorator to count number of atoms before and after removing overlaps."""
    def wrapper(*args, **kwargs):
        initial_count = len(args[0])
        result = func(*args, **kwargs)
        final_count = len(result)
        tqdm.write(f"       Initial number of atoms: {initial_count}")
        tqdm.write(f" Atoms after removing overlaps: {final_count}")
        return result
    return wrapper