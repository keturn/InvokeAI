from functools import reduce
from typing import TypeVar

import torch
from contextlib import contextmanager
from tqdm import tqdm


def _gather_modules(module: torch.nn.Module, prefix=''):
    # this logic borrowed from accelerate.utils.compile_regions
    modules = []
    if isinstance(module, torch.nn.ModuleList):
        # for a list of like modules, compile each to take advantage of the compilation cache.
        if all(isinstance(submodule, module[0].__class__) for submodule in module):
            for index, submodule in enumerate(module):
                modules.append((f'{prefix}.{index}', submodule))
        else:
            modules.append((prefix, module))
    elif module._modules:  # Non-leaf node
        for name, submodule in module.named_children():
            modules.extend(_gather_modules(submodule, prefix=f'{prefix}.{name}'))
    else:  # Leaf node
        modules.append((f'{prefix}.{module.__class__.__name__}', module))

    return modules


M = TypeVar('M', bound=torch.nn.Module)

def compile_regions(module: M, **compile_kwargs) -> M:
    """
    Performs regional compilation where we target repeated blocks of the same class and compile them sequentially to
    hit the compiler's cache. For example, in `GPT2LMHeadModel`, the repeated block/class is `GPT2Block`, and can be
    accessed as `model.transformer.h[0]`. The rest of the model (e.g. model.lm_head) is compiled separately.

    This allows us to speed up the compilation overhead / cold start of models like LLMs and Transformers in general.
    See https://pytorch.org/tutorials/recipes/regional_compilation.html for more details.

    This implementation differs from accelerate.utils.compile_regions in two ways:
    1. It gathers the complete list of regions before marking any of them for compilation.
    2. It uses `Module.compile` instead of swapping out each Module instance.

    Args:
        module (`torch.nn.Module`):
            The model to compile.
        **compile_kwargs:
            Additional keyword arguments to pass to `torch.compile()`.

    Returns:
        `torch.nn.Module`: The model compiled.
    """
    modules = _gather_modules(module, prefix=module.__class__.__name__)
    for name, submodule in modules:
        submodule.compile(**compile_kwargs)
    return module


def regional_compile_bisect(module: M, halves=tuple(), **compile_kwargs) -> M:
    modules = _gather_modules(module, prefix=module.__class__.__name__)
    modules = bisect_list(modules, halves)
    progress = tqdm(modules, desc="Marking for compilation", unit="module")
    for (name, submodule) in progress:
        progress.set_postfix_str(name)
        submodule.compile(**compile_kwargs)
    return module


def halve_list(lst, direction):
    if len(lst) <= 1:
        return lst
    mid = len(lst) // 2
    return lst[mid:] if direction else lst[:mid]


def bisect_list(items, directions):
    """
    Bisect a list based on a sequence of directions.

    Args:
        items: List of items to bisect
        directions: Sequence of 0s and 1s where:
                   0 = take lower half
                   1 = take upper half

    Returns:
        Sublist after applying all bisection directions
    """
    return reduce(halve_list, directions, items)


@contextmanager
def default_dtype(dtype: torch.dtype):
    orig_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(orig_dtype)


@contextmanager
def log_compilation_time(logger):
    times_before = torch._dynamo.utils.calculate_time_spent()
    yield
    times_after = torch._dynamo.utils.calculate_time_spent()
    logger(f"Time spent in torch.compile: {times_after['total_wall_time'] - times_before['total_wall_time']:.3f}s")


def is_module_fully_loaded(module: torch.nn.Module) -> bool:
    autocasters = [m for m in module.modules() if hasattr(m, 'is_device_autocasting_enabled')]
    return all(not m.is_device_autocasting_enabled() for m in autocasters)
