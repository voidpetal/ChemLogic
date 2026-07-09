def main(*args, **kwargs):
    """Thin redirect to chemlogic.experiments.main — requires ChemLogic[experiments]."""
    from chemlogic.experiments import main as _main

    return _main(*args, **kwargs)


__all__ = ["main"]
