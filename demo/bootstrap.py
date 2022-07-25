from collections.abc import Iterator, Sequence
from random import randrange
from typing import TypeVar

T = TypeVar('T')


def bootstrap(items: Sequence[T], n: int | float) -> Iterator[T]:
    num_items = len(items)

    if n < 1:
        n = int(n * num_items)

    while n > 0:
        i = randrange(num_items)
        yield items[i]
        n -= 1


def main():
    file_in = 'in/wegingen-uit-excel-voor-bootstrap.csv'
    print(f'Lees regels uit {file_in!r}.')
    with open(file_in, 'r') as f:
        header = f.readline()
        data = f.readlines()
    print(f'Sample size is {len(data)}.')

    for i in range(1, 11):
        file_out = f'out/wegingen-bootstrap-{i:02d}.csv'
        print(f'Schrijf bootstrat {i} naar {file_out!r}.')
        with open(file_out, 'w') as f:
            f.write(header)
            f.writelines(bootstrap(data, 0.9))


if __name__ == '__main__':
    main()
