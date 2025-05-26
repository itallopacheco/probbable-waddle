from abc import ABC, abstractmethod
from pathlib import Path

import sys

PROJ_ROOT = Path(__file__).parents[2] 
sys.path.insert(0, str(PROJ_ROOT / "src"))

class PipelineStep(ABC):
    @abstractmethod
    def execute(self, context: dict) -> dict:
        ...
