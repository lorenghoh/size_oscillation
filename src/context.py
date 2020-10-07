from contextlib import contextmanager


@contextmanager
def add_path(p):
    import sys
    old_path = sys.path
    old_modules = sys.modules
    sys.modules = old_modules.copy()
    sys.path = sys.path[:]
    sys.path.insert(0, p)

    try:
        yield
    finally:
        sys.path = old_path
        sys.modules = old_modules
