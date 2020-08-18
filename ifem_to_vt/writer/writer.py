from abc import ABC, abstractmethod
from inspect import isabstract
from pathlib import Path

import treelog as log

from typing import Any, Optional, Dict
from ..typing import Array2D

from .. import config
from ..geometry import Patch, GeometryManager
from ..fields import FieldPatch, SimpleFieldPatch, CombinedFieldPatch
from ..util import subclasses



class Writer(ABC):

    writer_name: str

    geometry: GeometryManager
    outpath: Path

    stepid: int
    stepdata: Dict[str, Any]
    step_finalized: bool
    geometry_finalized: bool

    @classmethod
    def applicable(self, fmt: str) -> bool:
        """Return true if the class can handle the given format."""
        return False

    @staticmethod
    def find_applicable(fmt: str) -> type:
        """Return a writer subclass that can handle the given format."""
        for cls in subclasses(Writer, invert=True):
            if isabstract(cls):
                continue
            if cls.applicable(fmt):
                log.info(f"Using writer: {cls.writer_name}")
                return cls
            else:
                log.debug(f"Rejecting writer: {cls.writer_name}")
        raise TypeError(f"Unable to find any applicable writers for {fmt}")

    def __init__(self, outpath: Path):
        self.geometry = GeometryManager()
        self.outpath = Path(outpath)

    def validate(self):
        """Raise an error if config options are invalid."""
        pass

    def make_filename(self, root: Optional[Path] = None, with_step: bool = False, indexing: int = 1):
        """Create a filename based on the output path, including step ID or
        not, if required.  If ROOT is not given, the configured output
        path is used.
        """
        if root is None:
            root = self.outpath
        if not (with_step and config.multiple_timesteps):
            return root
        return root.with_name(f'{root.stem}-{self.stepid + indexing}').with_suffix(root.suffix)

    def __enter__(self):
        self.stepid = -1
        self.stepdata = dict()
        self.step_finalized = True
        self.geometry_finalized = True
        return self

    def __exit__(self, tp, value, bt):
        pass

    def add_step(self, **stepdata: Any):
        """Increment the step counter and store the data (which may be time,
        frequency, etc.)
        """
        assert self.step_finalized
        self.stepid += 1
        self.stepdata = stepdata
        self.step_finalized = False
        self.geometry_finalized = False

    def update_geometry(self, patch: Patch):
        """Call this after add_step to update the geometry for each new patch.
        This method only returns the global patch ID.  It should be
        reimplemented in subclasses.
        """
        assert not self.geometry_finalized
        return self.geometry.update(patch)

    def finalize_geometry(self):
        """Call this method after all patches have been updated to allow the
        writer to push data to disk.
        """
        assert not self.geometry_finalized
        self.geometry_finalized = True

    def update_field(self, field: FieldPatch):
        """Call this method after finalize_geometry to issue updates to fields
        which are defined on patches.  This method only returns the
        global patch ID.  It should be reimplemented in subclasses.
        """
        assert self.geometry_finalized
        assert not self.step_finalized

        if isinstance(field, SimpleFieldPatch):
            return self.geometry.global_id(field.patch)
        if isinstance(field, CombinedFieldPatch):
            patchids = {self.geometry.global_id(patch) for patch in field.patches}
            assert len(patchids) == 1
            return next(iter(patchids))
        assert False

    def finalize_step(self):
        """Call this method after all calls to update_field have been issued,
        to allow the writer to push data to disk.
        """
        assert self.geometry_finalized
        assert not self.step_finalized
        self.step_finalized = True
