from abc import ABC, abstractmethod
from dataclasses import dataclass
from attrs import define


class Test(ABC):
    @property
    @abstractmethod
    def dong() -> int:
        ...


@dataclass
class YourTest(Test):
    dong: int


@define
class MyTest(Test):
    dong: int


print(YourTest.__dict__)
print(MyTest.__dict__)


q = MyTest(dong=3)
print(q)
print(q.dong)

p = YourTest(dong=3)
print(p)
print(p.dong)
