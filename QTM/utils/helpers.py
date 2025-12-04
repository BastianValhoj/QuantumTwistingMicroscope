
def in_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell" # if its a notebook, this comparison will return true
    except Exception:
        return False