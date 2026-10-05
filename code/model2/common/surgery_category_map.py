"""SNOMED procedure code -> coarse surgery category (from check_prediction.ipynb)."""
from __future__ import annotations

from typing import Any, Dict, Set

SNOMED_TO_SURGERY_CATEGORY: Dict[str, str] = {
    "SNOMED|11466000": "Abdomen",
    "SNOMED|45595009": "Abdomen",
    "SNOMED|6025007": "Abdomen",
    "SNOMED|74770008": "Abdomen",
    "SNOMED|73632009": "Abdomen",
    "SNOMED|427074001": "Abdomen",
    "SNOMED|116141005": "Abdomen",
    "SNOMED|20630000": "Abdomen",
    "SNOMED|90199006": "Abdomen",
    "SNOMED|703945007": "Abdomen",
    "SNOMED|386791007": "Abdomen",
    "SNOMED|77543007": "Abdomen",
    "SNOMED|15018006": "Abdomen",
    "SNOMED|1137453008": "Abdomen",
    "SNOMED|235159007": "Abdomen",
    "SNOMED|609230000": "Abdomen",
    "SNOMED|1156322007": "Abdomen",
    "SNOMED|438341004": "Abdomen",
    "SNOMED|8324006": "Abdomen",
    "SNOMED|708877008": "Abdomen",
    "SNOMED|236887006": "Abdomen",
    "SNOMED|265636007": "Abdomen",
    "SNOMED|314128006": "Abdomen",
    "SNOMED|116144002": "Abdomen",
    "SNOMED|708919000": "Abdomen",
    "SNOMED|44558001": "Abdomen",
    "SNOMED|24496007": "Abdomen",
    "SNOMED|107944001": "Abdomen",
    "SNOMED|1390003": "Abdomen",
    "SNOMED|773225002": "Abdomen",
    "SNOMED|608890000": "Abdomen",
    "SNOMED|11401008": "Abdomen",
    "SNOMED|75835007": "Abdomen",
    "SNOMED|23140002": "Abdomen",
    "SNOMED|440118004": "Abdomen",
    "SNOMED|712992007": "Abdomen",
    "SNOMED|708627007": "Abdomen",
    "SNOMED|359887003": "Abdomen",
    "SNOMED|238298006": "Abdomen",
    "SNOMED|42010004": "Abdomen",
    "SNOMED|771708006": "Abdomen",
    "SNOMED|44946007": "Abdomen",
    "SNOMED|608833000": "Abdomen",
    "SNOMED|444588006": "Abdomen",
    "SNOMED|25353009": "Intracranial",
    "SNOMED|47020004": "Intracranial",
    "SNOMED|32495000": "Intracranial",
    "SNOMED|36910002": "Intracranial",
    "SNOMED|719351007": "Thoracic",
    "SNOMED|58380004": "Thoracic",
    "SNOMED|22418005": "Thoracic",
    "SNOMED|319175008": "Thoracic",
    "SNOMED|59214008": "Thoracic",
    "SNOMED|77202008": "Thoracic",
    "SNOMED|26337002": "Thoracic",
    "SNOMED|232717009": "Cardiac",
    "SNOMED|119564002": "Cardiac",
    "SNOMED|609588000": "Joint",
    "SNOMED|52734007": "Joint",
    "SNOMED|309431009": "Joint",
    "SNOMED|699120002": "Joint",
    "SNOMED|42262007": "Joint",
    "SNOMED|8874001": "Joint",
    "SNOMED|37422000": "Joint",
    "SNOMED|280462001": "Joint",
    "SNOMED|608809006": "Joint",
    "SNOMED|281812007": "Joint",
    "SNOMED|62402000": "Joint",
    "SNOMED|386682000": "Joint",
    "SNOMED|66339000": "Joint",
    "SNOMED|438557003": "Spine",
    "SNOMED|449378007": "Spine",
    "SNOMED|709408000": "Spine",
    "SNOMED|359586003": "Spine",
    "SNOMED|3418002": "Spine",
    "SNOMED|830198007": "Spine",
    "SNOMED|426243004": "Spine",
    "SNOMED|265675003": "Skin",
    "SNOMED|302405008": "Skin",
}


def surgery_categories_for_episode(episode: Dict[str, Any]) -> Set[str]:
    cats: Set[str] = set()
    for tok in episode.get("procedure_tokens", []) or []:
        code = tok.get("code") if isinstance(tok, dict) else tok
        if code and str(code) in SNOMED_TO_SURGERY_CATEGORY:
            cats.add(SNOMED_TO_SURGERY_CATEGORY[str(code)])
    return cats


def primary_surgery_type(episode: Dict[str, Any], default: str = "Unknown") -> str:
    cats = sorted(surgery_categories_for_episode(episode))
    return cats[0] if cats else default
