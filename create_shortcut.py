from pyshortcuts import make_shortcut
import sys
import os

def create_shortcut():
    # Get the path to the activated environment's napari
    napari_path = os.path.join(os.path.dirname(sys.executable), 'napari')
    make_shortcut(
        f'{napari_path} -w lumin',
        name='LUMIN',
        terminal=False,
        desktop=True,
        startmenu=True
    )

if __name__ == '__main__':
    create_shortcut()