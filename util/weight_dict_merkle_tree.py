from typing import Dict

from torch import Tensor


class WeightDictMerkleTree:

    def __init__(self, weight_dict: Dict[str, Tensor]):
        pass

    def to_dict(self) -> dict:
        pass

    @classmethod
    def from_dict(cls, hash_info_dict):
        pass

    def __eq__(self, other):
        # to implement this method we only have to compare the root hash
        return False

    def __hash__(self):
        # here we can just return the root hash
        return hash(self)
