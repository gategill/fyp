"""

"""


from time import sleep
from icecream import ic
from dataset.Dataset import Dataset


class CoRecRecommender:
    def __init__(self, k: int) -> None:
        ic("cr_rec.__init__()")
