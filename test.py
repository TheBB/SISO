from dataclasses import dataclass
import click


class SuperType: ...

@dataclass
class ByNumber(SuperType):
    value: int

@dataclass
class ByName(SuperType):
    value: str


def store_number(ctx, param, value):
    if value is not None:
        ctx.params['obj'] = ByNumber(value)
    return value


def store_name(ctx, param, value):
    if value is not None:
        ctx.params['obj'] = ByName(value)
    return value


@click.command()
@click.option('--number', type=int, expose_value=False, callback=store_number)
@click.option('--name', type=str, expose_value=False, callback=store_name)
def main(obj):
    print(repr(obj))


main()
