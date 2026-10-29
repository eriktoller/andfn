from importlib import import_module

PARALLEL = True
CACHE = True


def __getattr__(name):
    try:
        module = import_module(f".{name}", __name__)
    except ModuleNotFoundError as exc:
        expected = f"{__name__}.{name}"

        if exc.name != expected:
            raise

        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from None

    globals()[name] = module
    return module
