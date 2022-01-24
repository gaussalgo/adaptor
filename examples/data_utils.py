from abc import abstractmethod, ABC
from enum import Enum
from typing import List


class DataSplit(Enum):
    train = 0
    val = 1
    test = 2
    end = 3


class SplitOffset(Enum):
    train = 0.0
    val = 0.8
    test = 0.9
    end = 1.0


class DataSource(ABC):
    dataset_id: str = None

    @abstractmethod
    def __init__(self,  split: str, data_dir: str):
        self.split = split
        self.data_dir = data_dir

    def _get_in_split(self, data: List) -> List:
        offset_start_pos = int(len(data) * SplitOffset[self.split].value)
        offset_end_pos = int(len(data) * list(SplitOffset)[DataSplit[self.split].value + 1].value)
        return data[offset_start_pos: offset_end_pos]
