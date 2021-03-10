from enum import Enum


class ModelStoreType(Enum):
    PICKLED_WEIGHTS = '1'
    WEIGHT_UPDATES = '2'
    PROVENANCE = '3'
