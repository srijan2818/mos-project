"""One process-wide Gmsh session shared by the CAD importer, generators and image meshers.

Gmsh keeps global state and is not thread-safe, so every caller that builds or meshes a
model must hold :data:`GMSH_LOCK`. ``gmsh.initialize()`` by default installs a SIGINT
handler, which raises ``ValueError`` outside the main thread; :func:`ensure_gmsh`
initialises without it so background workers in the app can use Gmsh too.
"""
from __future__ import annotations

import threading

try:
    import gmsh
    HAS_GMSH = True
    GMSH_ERROR = ""
except (ImportError, OSError) as _exc:   # OSError: installed but the shared library won't load
    gmsh = None
    HAS_GMSH = False
    GMSH_ERROR = f"{type(_exc).__name__}: {_exc}"

GMSH_LOCK = threading.RLock()


def ensure_gmsh() -> None:
    """Initialise Gmsh once (thread-safe, no signal handler) and silence its terminal output."""
    if not HAS_GMSH:
        raise RuntimeError("Gmsh is required but could not be loaded"
                           + (f" ({GMSH_ERROR})" if GMSH_ERROR else "")
                           + ". Install it with: conda install -c conda-forge python-gmsh")
    with GMSH_LOCK:
        if not gmsh.isInitialized():
            try:
                gmsh.initialize(interruptible=False)
            except TypeError:          # gmsh < 4.11 has no `interruptible` argument
                gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 1)
