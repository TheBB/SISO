import logging
from pathlib import Path

from typing import Callable, List, Tuple, Optional

from ..api import Source


ReaderCheck = Callable[[Path], Optional[Source]]

READERS: List[Tuple[str, ReaderCheck]] = []

def register_reader(name: str):
    def inner(check: ReaderCheck):
        READERS.append((name, check))
    return inner

def find_reader(path: Path) -> Optional[Source]:
    for name, reader in READERS:
        try:
            source = reader(path)
        except ImportError:
            logging.debug(f'Unable to check for {name} format - some dependencies may not be installed')
            continue
        if not source:
            logging.debug(f'File appears not to be {name} format - skipping')
            continue
        logging.info(f'Type of {path} determined to be {name}')
        return source
    return None


@register_reader('GoTools')
def gotools(path: Path) -> Optional[Source]:
    if path.suffix.lower() != '.g2':
        return None
    from .gotools import GoTools
    return GoTools(path)


@register_reader('IFEM')
def ifem(path: Path) -> Optional[Source]:
    from .ifem import Ifem
    if Ifem.applicable(path):
        return Ifem(path)
    return None
