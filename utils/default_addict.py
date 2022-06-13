from addict import addict


class Default_Addict(addict.Dict):
    def __missing__(self, key):
        return 0

