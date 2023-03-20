import os

def list_dirs(path):
    elts = os.listdir(path)
    out = []
    for e in elts:
        if os.path.isdir(os.path.join(path, e)):
            out.append(e)
    return out