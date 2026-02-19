from dataclasses import dataclass, asdict

@dataclass
class TrainConfig:
    data_dir: str = "data/raw"
    output_dir: str = "models"
    val_weeks: int = 6
    seed: int = 10

cfg1 = TrainConfig(seed=10)
cfg2 = TrainConfig(seed=10)

print(cfg1 == cfg2)   # True
print(cfg1)