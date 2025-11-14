from time import time


def timeit(func):
    """Decorator to time function call."""
    def wrapper(*args, **kwargs):
        initial_time = time()
        result = func(*args, **kwargs)
        final_time = time()
        func_args = {}
        # if (args is None) or (len(args) == 0):
        #     func_args["width"] = kwargs["width"]
        #     func_args["length"] = kwargs["length"]
        # else:
        #     func_args["width"] = args[0]
        #     func_args["length"] = args[1]
        time_diff = abs(final_time - initial_time)
        hours, rem = divmod(time_diff, 60*60)
        minutes, seconds = divmod(rem, 60)
        print(f"func '{func.__name__}' with args : {"-not implemented-"}  ran in {hours:02}:{minutes:02}:{seconds:05.3f}")
        return result
    return wrapper

def count_removals(func):
    """Decorator to count number of atoms before and after removing overlaps."""
    def wrapper(*args, **kwargs):
        initial_count = len(args[0])
        result = func(*args, **kwargs)
        final_count = len(result)
        print(f"       Initial number of atoms: {initial_count}")
        print(f" Atoms after removing overlaps: {final_count}")
        return result
    return wrapper