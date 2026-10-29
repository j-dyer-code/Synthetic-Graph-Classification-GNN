import copy
import json
from pathlib import Path
from lib.common import file_hash

ROOT = Path(__file__).resolve().parent


def package_identity():
    paths = sorted(
        [ROOT / "protocol.json", *ROOT.glob("*.py"), *(ROOT / "lib").glob("*.py")]
    )
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}


def load_protocol(software_test=False):
    protocol = json.loads((ROOT / "protocol.json").read_text())
    if software_test:
        protocol = copy.deepcopy(protocol)
        protocol["purpose"] = protocol["main"]["purpose"] = "software_test"
        protocol["seeds"] = protocol["main"]["seeds"] = [42]
        protocol["main"]["data"]["graphs_per_family"] = 10
        protocol["main"]["tuning"].update(
            trials=2,
            epochs=2,
            objective="validation loss after two epochs; software test only",
        )
        protocol["main"]["training"].update(max_epochs=2, patience=2)
        protocol["ablation"].update(max_epochs=2, patience=2)
    return protocol
