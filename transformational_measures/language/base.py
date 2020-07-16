import abc

class Language(abc.ABC):
    def __init__(self):
        self.name = ""
        self.layer = ""
        self.layers = ""
        self.transformation = ""
        self.transformations = ""
        self.meandeviation = ""

        self.measure=""

        self.stratified=""
        self.non_stratified=""

        self.samples = ""
        self.model=""
        self.mean=""
        self.max=""
        self.min=""
        self.sum=""

