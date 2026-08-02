"""Backend selector — import and return the requested ops module."""


def get_backend(name: str):
    """Return the ops module for `name` {'python', 'c', 'asm'}.
    Phase 1: only 'python' is implemented.
    """
    if name == "python":
        from ops import backend_python as mod
        return mod
    if name == "c":
        from ops import backend_c as mod
        return mod
    if name == "asm":
        raise NotImplementedError(
            "Assembly backend not yet implemented (phase 3). Use --backend python."
        )
    raise ValueError(f"Unknown backend: {name!r}. Choose from: python, c, asm.")
