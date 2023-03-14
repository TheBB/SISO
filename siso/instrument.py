from __future__ import annotations

from typing import Callable, Dict, Optional, Protocol, Tuple

from attrs import define

from .api import Source


class Method(Protocol):
    __func__: Callable
    __name__: str


METHODS = [
    "field_data",
    "topology",
]


class MethodInstrumenter:
    func: Method

    def __init__(self, func: Method):
        self.func = func
        self.call_data = CallData()

    def __call__(self, *args, **kwargs):
        self.call_data.ncalls += 1
        return self.func(*args, **kwargs)

    def format(self) -> str:
        desc = f"{self.func.__name__}: {self.call_data.ncalls} calls"
        func = self.func.__func__
        if hasattr(func, "cache_info"):
            info = func.cache_info()
            hitrate = info.hits / (info.hits + info.misses)
            desc = f"{desc} ({hitrate*100:.2g}% cache hit rate)"
        return desc


@define
class CallData:
    ncalls: int = 0


class Instrumenter:
    original_source: Source
    sources: Dict[int, Source]
    instrumenters: Dict[Tuple[int, str], MethodInstrumenter]

    def __init__(self, source: Source):
        self.original_source = source
        self.sources = {}
        self.instrumenters = {}
        self.discover_sources(source)
        self.plug_instrumenters()

    def discover_sources(self, source: Source):
        self.sources[id(source)] = source
        for src in source.children():
            self.discover_sources(src)

    def plug_instrumenters(self):
        for source in self.sources.values():
            for name in METHODS:
                instrumenter = MethodInstrumenter(getattr(source, name))
                self.instrumenters[(id(source), name)] = instrumenter
                setattr(source, name, instrumenter)

    def report(self, sprefix: str = "", prefix: str = "", source: Optional[Source] = None):
        source = source or self.original_source
        children = list(source.children())
        line = "│" if children else " "

        print(f"{sprefix}{source.__class__.__name__}")

        for name in METHODS:
            instrumenter = self.instrumenters[(id(source), name)]
            print(f"{prefix}{line}  {instrumenter.format()}")

        if not children:
            return

        print(f"{prefix}{line}")
        if len(children) == 1:
            self.report(prefix, prefix, children[0])
        else:
            for c in children[:-1]:
                self.report(f"{prefix}├─ ", f"{prefix}│  ", c)
                print("│")
            self.report(f"{prefix}└─ ", f"{prefix}   ", children[-1])
