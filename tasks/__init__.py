from invoke import Collection

from . import classifier, tool, metrics, test

namespace = Collection(
    classifier,
    tool,
    metrics,
    test
)
