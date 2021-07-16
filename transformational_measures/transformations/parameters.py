import numpy as np
import abc
import itertools
from typing import Iterator

class TransformationParameters(abc.ABC,list):
    def __init__(self,members:[]):
        super().__init__(members)

    @abc.abstractmethod
    def id(self)->str:
        pass

    def __repr__(self)->str:
        return self.id()

    def scale3(self, scales, include_identity=False):
        result=[]
        if include_identity:
            result.append((1,1))
        for s in scales:
            if s == 1 and include_identity:
                continue
            s2 = [(1,s),(s,1),(s,s)]
            result+=s2
        return result

    def scale1(self, scales, include_identity=False):
        result=[]
        if include_identity:
            result.append((1,1))
        result+= [(s,s) for s in scales if not (s==1 and include_identity)]
        return result

    def translations8(self, values, include_identity=False):
        translations=[]
        if include_identity:
            translations.append([0,0])

        for d in values:
            p=[0,d,-d]
            combinations = list(itertools.product(p,p))
            combinations=combinations[1:]# remove (0,0)
            translations+=combinations
        return translations

    def translations4(self, values, include_identity=False):
        translations=[]
        if include_identity:
            translations.append([0,0])

        for d in values:
            translations+=[ (0,d),(d,0),(0,-d),(-d,0)] # clockwise NESW
        return translations

    def translations4diag(self, values, include_identity=False):
        translations=[]
        if include_identity:
            translations.append([0,0])

        for d in values:
            translations+=[ (-d,-d),(-d,d),(d,d),(d,-d)] #clockwise SW NW NE SE
        return translations

class ConstantParameters(TransformationParameters):
    def __init__(self,values:[]):
        super().__init__(values)
    def id(self):
        return f"C({self[:]})"


class UniformRotation(TransformationParameters):
    def __init__(self, n:int=1,angles:float=0):
        self.n=n
        if n<0: raise ValueError(f"Wrong value for n")
        if  angles <0 or angles >1: raise ValueError(f"Wrong angle range {angles}")
        self.angles=angles
        super().__init__(self.values())

    def values(self):
        if self.n==0:
            return [0]
        else:
            # TODO change to -self.angles to self.angles and use 180 as max
            # So that changes in angle correspond more closely to changes in
            # transformation "complexity"
            # In this way, both α and -α always belong to the same t set
            a=self.angles/2
            return np.linspace(-a, a, self.n)

    def id(self):
        return f"UR({self.angles},{self.n})"

class ScaleParameters(TransformationParameters):
    pass

class ScaleUniform(ScaleParameters):
    def __init__(self, n:int=0, min_downscale:float=1, max_upscale:float=1,include_identity=True):
        # generates n*2*3+1=n*6+1 scales
        # It generates n 1D downscales uniformly sampled from (min_downscale,1)
        # and n 1D upscales uniformly sampled from (1,max_upscale)
        # for n*2 total 1D scales
        # Then each those 1D scales are transformed into 3 2D scales such that:
        # s -> [ (1,s), (s,1), (s,s)]
        # The identity transform is also included
        self.n=n
        if  min_downscale <0 or min_downscale>1 or max_upscale<1: raise ValueError(f"Values must satisfy  0<{min_downscale}<=1<={max_upscale}")
        self.min_downscale, self.max_upscale= min_downscale, max_upscale
        self.include_identity=include_identity
        super().__init__(self.values())

    def values(self) ->[float,float]:
        if self.n==0:
            return (1,1)
        upscale = np.linspace(1,self.max_upscale,self.n+1,endpoint=True)
        upscale=upscale[1:]# remove scale=1
        downscale = np.linspace(self.min_downscale,1,self.n,endpoint=False)
        return self.scale3(downscale)+self.scale3(upscale,include_identity=self.include_identity)

    def id(self):
        return f"US({self.min_downscale}-{self.max_upscale},{self.n})"

class ScaleUniformSymmetric(ScaleUniform):
    def __init__(self,n:int=0,intensity:float=0):
        super().__init__(n,1-intensity,1+intensity)

class TranslationUniform(TransformationParameters):
    def __init__(self, n:int=1,max_intensity:float=0,):
        assert max_intensity >= 0 and max_intensity <= 100
        self.n=n
        self.max_intensity=max_intensity
        super().__init__(self.values())

    def values(self):
        if self.n==0:
            return [(0,0)]
        values=np.linspace(0, self.max_intensity, self.n+1,endpoint=True)
        values=values[1:] # remove 0
        return self.translations8(values,include_identity=True)
    def id(self):
        return f"UT({self.max_intensity},{self.n})"


class NoRotation(TransformationParameters):
    def __init__(self):
        super().__init__([0])
    def id(self):
        return f"NR"

class NoScale(TransformationParameters):
    def __init__(self):
        super().__init__([(1,1)])
    def id(self):
        return f"NS"

class NoTranslation(TransformationParameters):
    def __init__(self):
        super().__init__([(0,0)])
    def id(self):
        return f"NT"
