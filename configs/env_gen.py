from dataclasses import dataclass


@dataclass
class EnvGeneration:
    # How many updates to perform on the path-finding model to perform before generating new environments.
    gen_interval: int = 200
    evo_batch_size: int = 64

