"""`requests` as the installer's Python finds it in a test that starts one (test_install_sh.py and
common/install/test_main.py put this folder first on its path): what those processes run reaches
no network, importing the real package was most of their time, and a step that did reach for it
fails here by name."""


def __getattr__(name):
    if name.startswith('__'):
        raise AttributeError(name)
    raise RuntimeError(f'requests.{name}: a script test of the installer reached for the network')
