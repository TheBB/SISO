from contextlib import contextmanager
from enum import IntEnum, Enum

from typing import Dict, Any, Optional, Tuple

import treelog as log


class ConfigSource(IntEnum):
    """Simple enum that tracks where a config value comes from, in
    order of 'overridability'.
    """

    # Default values. Can be changed at will when needed.
    Default = 0

    # Values provided by the user. We can change these if needed but
    # must issue a warning when doing so.
    User = 1

    # Values required by the program. We cannot change these if
    # needed: an error must be thrown instead.
    Required = 2

Default = ConfigSource.Default
User = ConfigSource.User
Required = ConfigSource.Required


class ConfigTarget(Enum):
    """Simple enum that tracks the intended target of a config option."""

    # Option intended for readers
    Reader = 0

    # Option intended for writers
    Writer = 1

Reader = ConfigTarget.Reader
Writer = ConfigTarget.Writer



class Config:
    """Configuration object that should be shared between reader and
    writer.  It is initialized by the main function, but may have its
    attributes manipulated by either reader or writer depending on
    their needs.
    """

    # Number of subdivisions for additional resolution.
    nvis = 1

    # Whether the data set contains multiple time steps.
    multiple_timesteps = True

    # Whether to copy only the final time step.
    only_final_timestep = False

    # Output mode. Used by the VTK and VTF writers.
    output_mode = 'binary'

    # Input endianness indicator. Used by the SIMRA reader.
    input_endianness = 'native'

    # List of basis objects to copy to output. Used by the IFEM
    # reader.
    only_bases = ()

    # Which basis should be used to represent the geometry. Used by
    # the IFEM reader.
    geometry_basis = None

    # Volumetric/surface field behaviour. Used by the WRF reader.
    # - volumetric: only include volumetric fields
    # - planar: only include surface fields
    # - extrude: include all, extruding surface fields upwards
    volumetric = 'volumetric'

    # Global or local mapping behaviour. Used by the WRF reader.
    mapping = 'local'

    # Hint to the reader that the data may be periodic. Used by the
    # WRF reader.
    periodic = False

    # Keeps track of which keys come from which source
    _value_sources: Dict[str, ConfigSource]

    # Keeps track of which intended targets each option has
    _targets: Dict[str, Tuple[ConfigTarget]] = {
        'only_final_timestep': (Reader,),
        'input_endianness': (Reader,),
        'only_bases': (Reader,),
        'geometry_basis': (Reader,),
        'volumetric': (Reader,),
        'mapping': (Reader,),
        'periodic': (Reader,),

        'output_mode': (Writer,),
        'multiple_timesteps': (Writer,),
    }

    def __init__(self):
        self._value_sources = dict()

    def source(self, key: str) -> ConfigSource:
        """Get the source of a setting."""
        return self._value_sources.get(key, ConfigSource.Default)

    def target_compatible(self, key: str, target: ConfigTarget) -> bool:
        """Check if a setting is intended for a target."""
        return target in self._targets.get(key, ())

    def upgrade_source(self, key: str, source: ConfigSource):
        """Override the source of a setting. This will only ever increase the
        source level, and will do so quietly: no warnings will be
        issued.
        """
        current_source = self.source(key)
        if current_source > source:
            raise ValueError(f"Attempted to downgrade source of {key}")
        self._value_sources[key] = source

    def assign(self, key: str, value: Any, source: ConfigSource, reason: Optional[str] = None):
        """Assign a value to a setting.  This may issue a warning or
        an error in case the source levels require it.
        """
        current_source = self.source(key)
        current_value = getattr(self, key)
        if current_source == ConfigSource.Required and value != current_value:
            if reason is not None:
                raise ValueError(f"Incompatibility with setting '{key}': {reason}")
            else:
                raise ValueError(f"Incompatibility with setting '{key}'")

        elif current_source == ConfigSource.User and value != current_value:
            log.warning(f"Setting '{key}' was overridden")
            if reason is not None:
                log.warning(f"Reason: {reason}")

        assert current_source <= source
        setattr(self, key, value)
        self._value_sources[key] = source

    def require(self, reason: Optional[str] = None, **kwargs: Any):
        """Set some settings to new values with the highest source
        level.  If they have previously been required to have
        different values, an error will be thrown.
        """
        for key, value in kwargs.items():
            self.assign(key, value, ConfigSource.Required, reason)

    def ensure_limited(self, target: ConfigTarget, *args: str,
                       reason: Optional[str] = None,
                       lower: ConfigSource = ConfigSource.User,
                       upper: ConfigSource = ConfigSource.User):
        """Ensure that the set of explicitly sourced settings for a given
        target is limited to only those given.  If not, an error will
        be thrown.  By default, checks the user-provided settings.
        """
        args = set(args)
        for key, source in self._value_sources.items():
            if self.target_compatible(key, target) and lower <= source <= upper and key not in args:
                if reason is not None:
                    raise ValueError(f"'{key}' should not have been set ({reason})")
                else:
                    raise ValueError(f"'{key}' should not have been set")

    @contextmanager
    def __call__(self, source: ConfigSource = ConfigSource.User, **kwargs: Any):
        """Context manager for running code with different settings."""
        prev = dict(self.__dict__)
        prev_sources = dict(self._value_sources)
        for key, value in kwargs.items():
            self.assign(key, value, source)
        try:
            yield
        finally:
            self.__dict__ = prev
            self._value_sources = prev_sources


config = Config()
