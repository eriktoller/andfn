"""Numeric record views over existing fracture metadata and CSR connectivity."""

import operator
from collections import namedtuple

import numba as nb
from numba.extending import overload, overload_attribute

_Row = namedtuple("_Row", ("metadata", "ids"))


class FractureRecord(_Row):
    __slots__ = ()

    def __getitem__(self, key):
        if key == "elements":
            return self.ids
        if key == "nelements":
            return self.ids.size
        return self.metadata[0][key]

    def __setitem__(self, key, value):
        self.metadata[0][key] = value


_Arrays = namedtuple("_Arrays", ("metadata", "ids", "indptr"))


class FractureArrays(_Arrays):
    __slots__ = ()

    @property
    def size(self):
        return self.metadata.size

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.metadata[key]
        start, end = self.indptr[key : key + 2]
        return FractureRecord(self.metadata[key : key + 1], self.ids[start:end])


@overload(operator.getitem)
def _getitem(value, key):
    if (
        isinstance(value, nb.types.NamedTuple)
        and value.instance_class is FractureArrays
    ):
        if isinstance(key, nb.types.StringLiteral):
            name = key.literal_value

            def column(value, key):
                return value.metadata[name]

            return column

        def row(value, key):
            start, end = value.indptr[key], value.indptr[key + 1]
            return FractureRecord(value.metadata[key : key + 1], value.ids[start:end])

        return row
    if (
        isinstance(value, nb.types.NamedTuple)
        and value.instance_class is FractureRecord
        and isinstance(key, nb.types.StringLiteral)
    ):
        name = key.literal_value
        if name == "elements":

            def ids(value, key):
                return value.ids

            return ids
        if name == "nelements":

            def size(value, key):
                return value.ids.size

            return size

        def scalar(value, key):
            return value.metadata[0][name]

        return scalar


@overload(operator.setitem)
def _setitem(value, key, item):
    if (
        isinstance(value, nb.types.NamedTuple)
        and value.instance_class is FractureRecord
        and isinstance(key, nb.types.StringLiteral)
    ):
        name = key.literal_value

        def scalar(value, key, item):
            value.metadata[0][name] = item

        return scalar


@overload_attribute(nb.types.NamedTuple, "size")
def _size(value):
    if value.instance_class is FractureArrays:

        def size(value):
            return value.metadata.size

        return size
