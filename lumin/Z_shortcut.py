import os
import sys


def create_shortcut():
    """
    Creates a Windows desktop shortcut to run LUMIN in the correct environment.
    """

    try:
        from pyshortcuts import make_shortcut

        # Use CLI entry point (IMPORTANT: NOT file path)
        script = "lumin"

        icon_path = None  # optional: add icon later

        make_shortcut(
            script=script,
            name="LUMIN",
            desktop=True,
            startmenu=True,
            icon=icon_path,
        )

        print("[LUMIN] Desktop shortcut created successfully.")
    except ImportError:
        print(
            "[LUMIN] pyshortcuts is not installed.\n"
            "Run: pip install pyshortcuts"
        )
    except Exception as e:
        print(f"[LUMIN] Shortcut creation failed: {e}")