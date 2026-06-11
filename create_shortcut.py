from pyshortcuts import make_shortcut
import os

def create_shortcut():
    bat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LUMIN.bat')
    make_shortcut(
        bat_path,
        name='LUMIN',
        terminal=True,
        desktop=True,
        startmenu=True
    )

if __name__ == '__main__':
    create_shortcut()