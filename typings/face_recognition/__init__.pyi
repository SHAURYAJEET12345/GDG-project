"""Type stub for face_recognition — covers only the functions used by this project."""

from typing import List, Tuple

import numpy as np


def load_image_file(file: str, mode: str = ...) -> np.ndarray: ...

def face_locations(
    img: np.ndarray,
    number_of_times_to_upsample: int = ...,
    model: str = ...,
) -> List[Tuple[int, int, int, int]]: ...

def face_encodings(
    face_image: np.ndarray,
    known_face_locations: List[Tuple[int, int, int, int]] | None = ...,
    num_jitters: int = ...,
    model: str = ...,
) -> List[np.ndarray]: ...

def face_distance(
    face_encodings: List[np.ndarray],
    face_to_compare: np.ndarray,
) -> np.ndarray: ...

def compare_faces(
    known_face_encodings: List[np.ndarray],
    face_encoding_to_check: np.ndarray,
    tolerance: float = ...,
) -> List[bool]: ...
