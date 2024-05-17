from dataclasses import dataclass
import numpy as np
from typing import List, Dict

@dataclass
class DiversityFilterParameters:
    name: str
    minscore: float = 0.4
    bucket_size: int = 25
    minsimilarity: float = 0.4


@dataclass
class ComponentParameters:
    component_type: str
    name: str
    weight: float
    specific_parameters: dict = None


@dataclass
class ComponentSummary:
    total_score: np.array
    parameters: ComponentParameters
    raw_score: np.ndarray = None


class FinalSummary:
    def __init__(self, total_score: np.array, scored_smiles: List[str],
                 scaffold_log_summary):
        self.total_score = total_score
        self.scored_smiles = scored_smiles
        self.valid_idxs = list(range(len(total_score)))




