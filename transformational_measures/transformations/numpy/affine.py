import cv2
import numpy as np
from skimage import transform as skimage_transform

from ..affine import AffineTransformation,AffineParameters


class AffineTransformationNumpy(AffineTransformation):
    def __init__(self,p:AffineParameters):
        super().__init__(p)
        self.transformation_matrix=self.generate_transformation_matrix(p)

    def generate_transformation_matrix(self, p:AffineParameters):
        transformation = skimage_transform.AffineTransform(scale=p.s, rotation=p.r*180, shear=None,translation=p.t)
        return transformation.params

    def center_transformation(self,transformation:skimage_transform.AffineTransform,image_size):
        h,w=image_size
        shift_y, shift_x = (h- 1) / 2., (w- 1) / 2.
        shift = skimage_transform.AffineTransform(translation=[-shift_x, -shift_y])
        shift_inv = skimage_transform.AffineTransform(translation=[shift_x, shift_y])
        return shift + (transformation+ shift_inv)

    def single(self,image:np.ndarray)->np.ndarray:
        input_shape=image.shape
        transformation= self.center_transformation(self.transformation_matrix,input_shape[:2])
        image_size=tuple(input_shape[:2])
        if input_shape[2] == 1:
            image = image[:, :, 0]
        image= cv2.warpPerspective(image, transformation, image_size)
        if input_shape[2]==1:
           image= image[:, :, np.newaxis]
        return image

    def __call__(self, batch:np.ndarray)->np.ndarray:

        results=[]
        for i in range(batch.shape[0]):
            x= batch[i, :]
            x=self.single(x)
            results.append(x)
        return np.stack(results, axis=0)

    def inverse(self):
        return AffineTransformationNumpy(self.ap.inverse())

    def numpy(self):
        return self