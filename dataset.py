from typing import List, Tuple


def load_file(file_path: str) -> Tuple[List[str], List[str]]:
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return [factors, expansions]
