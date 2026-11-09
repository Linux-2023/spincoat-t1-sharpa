"""One installed numerical task definition shared by the physical evaluator."""
import json
from pathlib import Path

CONTRACT = json.loads((Path(__file__).resolve().parents[1] / 'task_contract.json').read_text())
TASK_ID = CONTRACT['id']
CRITERIA = CONTRACT['acceptance']
DT = CONTRACT['physics_dt_ns'] / 1e9
