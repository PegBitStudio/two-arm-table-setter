"""A short, human-readable name for the Intel hardware this is running on — for video cards
and reports, so they never show the wrong laptop."""
import re

import openvino as ov


def _short(name: str) -> str:
    name = re.sub(r"\(R\)|\(TM\)|\(iGPU\)|\(dGPU\)|Processor|Graphics|@.*$", "", name)
    name = re.sub(r"^\s*\d+(st|nd|rd|th) Gen\s*", "", name)
    return re.sub(r"\s+", " ", name).strip()


def label() -> str:
    """e.g. 'Intel Core i7-1165G7 + Iris Xe' or 'Intel Core Ultra 7 258V + Arc 140V + NPU'."""
    core = ov.Core()
    devs = core.available_devices
    parts = [_short(core.get_property("CPU", "FULL_DEVICE_NAME"))]
    if "GPU" in devs:
        parts.append(_short(core.get_property("GPU", "FULL_DEVICE_NAME")).replace("Intel ", ""))
    if "NPU" in devs:
        parts.append("NPU")
    return " + ".join(parts)


if __name__ == "__main__":
    print(label())
