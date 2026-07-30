"""Backend selector — import and return the requested ops module."""


def get_backend(name: str):
    """Return the ops module for `name` {'python', 'c', 'asm'}.
    Phase 1: only 'python' is implemented.
    """
    if name == "python":
        from ops import backend_python as mod
        return mod
    if name == "c":
        raise NotImplementedError(
            "C backend not yet implemented (phase 2). Use --backend python."
        )
    if name == "asm":
        raise NotImplementedError(
            "Assembly backend not yet implemented (phase 3). Use --backend python."
        )
    raise ValueError(f"Unknown backend: {name!r}. Choose from: python, c, asm.")
