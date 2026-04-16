import argparse
import multiprocessing
import os
from pathlib import Path

from ._shortcut import create_shortcut

def first_run_check():
    flag_file = Path.home() / ".lumin_installed"

    if not flag_file.exists():
        create_shortcut()
        flag_file.write_text("installed")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--create-shortcut", action="store_true")

    args = parser.parse_args()

    if args.version:
        print("LUMIN v0.1.0")
        return

    if args.create_shortcut:
        create_shortcut()
        return

    # AUTO RUN ON FIRST INSTALL
    first_run_check()

    launch_napari()