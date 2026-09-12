"""
Windows MSVC runtime repair.

The problem
-----------
pandas 2.0.3 ships its own copy of the Visual C++ runtime beside one of its C
extensions:

    pandas/_libs/window/msvcp140.dll        14.29.30139.0
    pandas/_libs/window/vcruntime140_1.dll  14.29.30139.0

Wheels bundle these as a fallback for machines with no VC++ redistributable
installed. This machine has a much newer one (14.44), so the bundled copies are
not merely redundant, they are actively harmful.

Windows resolves DLL dependencies by base name, and the first module named
MSVCP140.dll loaded into a process wins for everything loaded afterwards.
Streamlit imports pandas during startup, so the stale 14.29 copy is loaded very
early. Libraries built against a newer toolchain - torch, sentencepiece,
onnxruntime, scikit-learn - then bind to it, look for symbols it does not have,
and the process dies instantly with access violation 0xc0000005.

That is a native crash, not a Python exception. There is no traceback and no
try/except can catch it. Under `streamlit run` it kills the server the moment a
browser connects and the script first touches those libraries, which looks like
the app "closing by itself" while the browser reports ERR_CONNECTION_REFUSED.

The fix
-------
Rename the stale bundled DLLs so Windows falls back to the system runtime in
System32. Run it after any `pip install`, which may restore them:

    python -m app._winruntime

Add --restore to undo. Nothing here does anything on non-Windows platforms,
where the whole problem does not exist.
"""

import os
import sys

# Runtime DLLs a wheel may bundle that can shadow the system copy.
_SHADOWING_DLLS = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")

# Packages known to bundle a stale runtime. Paths are relative to site-packages.
_KNOWN_BUNDLE_DIRS = (
    os.path.join("pandas", "_libs", "window"),
)

_DISABLED_SUFFIX = ".disabled"

_preloaded = False


# ----------------------------------------------------------------------
# Runtime preloading (defence in depth for non-Streamlit entry points)
# ----------------------------------------------------------------------


def preload_system_msvc_runtime() -> bool:
    """Load the system MSVC runtime ahead of any bundled copy.

    This helps only when it runs before the offending package is imported. For
    `streamlit run` that is impossible, because Streamlit imports pandas before
    the app script executes - which is why disable_stale_bundled_runtime() is
    the real fix. Still worth doing for plain scripts such as app/ingest.py.

    Safe to call repeatedly and on any platform.
    """
    global _preloaded
    if _preloaded:
        return True
    if not sys.platform.startswith("win"):
        _preloaded = True
        return False

    import ctypes

    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    loaded_any = False
    for name in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"):
        path = os.path.join(system32, name)
        if not os.path.exists(path):
            continue
        try:
            ctypes.WinDLL(path)
            loaded_any = True
        except OSError:
            # Not worth crashing over; whichever import needs it will report a
            # clearer error than we could here.
            pass

    _preloaded = True
    return loaded_any


# ----------------------------------------------------------------------
# The actual repair
# ----------------------------------------------------------------------


def _site_packages_dirs():
    seen = set()
    for entry in sys.path:
        if entry and entry.endswith("site-packages") and os.path.isdir(entry):
            real = os.path.realpath(entry)
            if real not in seen:
                seen.add(real)
                yield real


def find_stale_bundled_runtime():
    """Return paths of bundled runtime DLLs that shadow the system copy."""
    if not sys.platform.startswith("win"):
        return []

    found = []
    for site_dir in _site_packages_dirs():
        for rel in _KNOWN_BUNDLE_DIRS:
            bundle_dir = os.path.join(site_dir, rel)
            if not os.path.isdir(bundle_dir):
                continue
            for name in _SHADOWING_DLLS:
                path = os.path.join(bundle_dir, name)
                if os.path.isfile(path):
                    found.append(path)
    return found


def disable_stale_bundled_runtime(verbose: bool = True):
    """Rename shadowing DLLs so Windows uses the system runtime instead."""
    changed = []
    for path in find_stale_bundled_runtime():
        target = path + _DISABLED_SUFFIX
        if os.path.exists(target):
            os.remove(path)
        else:
            os.rename(path, target)
        changed.append(path)
        if verbose:
            print(f"disabled {path}")
    if verbose and not changed:
        print("Nothing to do: no stale bundled runtime DLLs found.")
    return changed


def restore_stale_bundled_runtime(verbose: bool = True):
    """Undo disable_stale_bundled_runtime()."""
    restored = []
    for site_dir in _site_packages_dirs():
        for rel in _KNOWN_BUNDLE_DIRS:
            bundle_dir = os.path.join(site_dir, rel)
            if not os.path.isdir(bundle_dir):
                continue
            for name in _SHADOWING_DLLS:
                disabled = os.path.join(bundle_dir, name + _DISABLED_SUFFIX)
                if os.path.isfile(disabled):
                    os.rename(disabled, os.path.join(bundle_dir, name))
                    restored.append(disabled)
                    if verbose:
                        print(f"restored {disabled}")
    if verbose and not restored:
        print("Nothing to restore.")
    return restored


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if not sys.platform.startswith("win"):
        print("Not Windows - nothing to do.")
        return 0

    if "--restore" in argv:
        restore_stale_bundled_runtime()
        print("\nRestored. The Streamlit crash will come back.")
        return 0

    changed = disable_stale_bundled_runtime()
    if changed:
        print(
            f"\nDisabled {len(changed)} stale runtime DLL(s). "
            "Streamlit should now stay up."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
