from dataclasses import dataclass


@dataclass
class EnvGeneration:
    # How many updates to perform on the path-finding model to perform before generating new environments.
    gen_interval: int = 200
    evo_batch_size: int = 64

@dataclass
class EnvGeneration100(EnvGeneration):
    gen_interval: int = 100
    evo_batch_size: int = 64

@dataclass
class EnvGeneration50(EnvGeneration):
    gen_interval: int = 50
    evo_batch_size: int = 64

@dataclass
class EnvGeneration10(EnvGeneration):
    gen_interval: int = 10
    evo_batch_size: int = 64

@dataclass
class EnvGeneration1(EnvGeneration):
    gen_interval: int = 1
    evo_batch_size: int = 64