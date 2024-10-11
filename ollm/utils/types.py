import os
from typing import TypeAlias, TypeVar

import networkx as nx

PathLike: TypeAlias = str | bytes | os.PathLike
Graph = TypeVar("Graph", bound=nx.Graph)
