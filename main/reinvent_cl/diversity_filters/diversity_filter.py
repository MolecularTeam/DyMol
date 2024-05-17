import abc
from component_summary import DiversityFilterParameters, ComponentSummary, ComponentParameters, FinalSummary
from diversity_filter_memory import DiversityFilterMemory
import pandas as pd
from typing import List, Dict
import numpy as np
from conversions import Conversions
from copy import deepcopy


class BaseDiversityFilter(abc.ABC):

    @abc.abstractmethod
    def __init__(self, parameters: DiversityFilterParameters):
        self.parameters = parameters
        self._diversity_filter_memory = DiversityFilterMemory()
        self._chemistry = Conversions()

    @abc.abstractmethod
    def update_score(self, score_summary: FinalSummary, step=0) -> np.array:
        raise NotImplementedError("The method 'update_score' is not implemented!")

    def filter_by_scaffold(self, smiles):
        survive_idx = []
        smiles_list = []
        scaffold_list = []
        for i, smile in enumerate(smiles):
            smile = self._chemistry.convert_to_rdkit_smiles(smile)
            scaffold = self._calculate_scaffold(smile)
            if self._smiles_exists(smile):
                continue
            if self._diversity_filter_memory.scaffold_instances_count(scaffold) > self.parameters.bucket_size:
                continue
            survive_idx.append(i)
            smiles_list.append(smile)
            scaffold_list.append(scaffold)
        return smiles_list, scaffold_list, survive_idx

    def add_with_filtered(self, score_summary: FinalSummary, scaffolds, step=0):
        score_summary = deepcopy(score_summary)
        scores = score_summary.total_score
        smiles = score_summary.scored_smiles
        for i, score in enumerate(scores):
            if score >= self.parameters.minscore:
                self._add_to_memory(i, score, smiles[i], scaffolds[i], score_summary, step)

    def get_memory_as_dataframe(self) -> pd.DataFrame:
        return self._diversity_filter_memory.get_memory()

    def set_memory_from_dataframe(self, memory: pd.DataFrame):
        self._diversity_filter_memory.set_memory(memory)

    def number_of_smiles_in_memory(self) -> int:
        return self._diversity_filter_memory.number_of_smiles()

    def number_of_scaffold_in_memory(self) -> int:
        return self._diversity_filter_memory.number_of_scaffolds()

    def update_bucket_size(self, bucket_size: int):
        self.parameters.bucket_size = bucket_size

    def _calculate_scaffold(self, smile):
        raise NotImplementedError

    def _smiles_exists(self, smile):
        return self._diversity_filter_memory.smiles_exists(smile)

    def _add_to_memory(self, indx: int, score, smile, scaffold, components, step):
        self._diversity_filter_memory.update(indx, score, smile, scaffold, components, step)

    def _penalize_score(self, scaffold, score):
        """Penalizes the score if the scaffold bucket is full"""
        if self._diversity_filter_memory.scaffold_instances_count(scaffold) > self.parameters.bucket_size:
            score = 0.
        return score






