# Copyright 2023 Kevin Turner
# SPDX-License-Identifier: Apache-2.0
"""Markov name generator.

InvokeAI wrapper for https://github.com/bicobus/pyMarkovNameGenerator

required dependencies: markovname~=1.0
"""
import json
import random
from importlib import resources

import markovname.data
from markovname import generator

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    FieldDescriptions,
    InputField,
    InvocationContext,
    invocation,
)
from invokeai.app.invocations.primitives import StringCollectionOutput, StringOutput
from invokeai.app.util.misc import SEED_MAX, get_random_seed

__version__ = "0.9.0"


def list_available_datasets() -> list[str]:
    return list(resources.contents(markovname.data))


def load_markovname_dataset(name: str) -> list[str]:
    with resources.open_text(markovname.data, name + ".json") as f:
        training_data = json.load(f)
    return training_data


def generate_name(training_data: list[str], order: int = 3, prior: float = 0, seed=0) -> str:
    rng = random.Random(seed)

    # TODO: add feature to upstream to set the seed or the RNG
    # for now, monkeypatch 🙊
    orig_random = generator.random
    generator.random = rng

    try:
        gen = markovname.Generator(training_data, order, prior)
        word = gen.generate().strip("#")
    finally:
        generator.random = orig_random

    return word


@invocation("markovname", title="Markov Name Generator", category="string", version=__version__)
class MarkovNameInvocation(BaseInvocation):
    """Generates random names."""

    seed: int = InputField(
        ge=0,
        le=SEED_MAX,
        description=FieldDescriptions.seed,
        default_factory=get_random_seed,
    )
    dataset: list[str] = InputField()
    order: int = InputField(
        default=3,
        description="Highest order of model to use. Will use Katz's back-off model. "
        "It looks for the next letter based on the last `n` letters.",
    )
    prior: float = InputField(default=0, gte=0, lte=1)

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(value=generate_name(self.dataset, self.order, self.prior, self.seed))


@invocation("markovname_loader", title="Markov Name Data Loader", category="string", version=__version__)
class MarkovNameLoaderInvocation(BaseInvocation):
    """Load data for the Markov Name Generator."""

    name: str = InputField()

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        return StringCollectionOutput(collection=load_markovname_dataset(self.name))
