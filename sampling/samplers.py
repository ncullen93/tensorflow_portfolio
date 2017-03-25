
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from . import BaseArraySampler, BaseDirectorySampler


class ImageArraySampler(BaseArraySampler):

    def __init__(self,
                x, 
                y,
                batch_size=32, 
                shuffle=True,
                augment=True,
                crop_shape=None,
                rotation_range=0.,
                shift_range=(0.,0.),
                shear_range=0.,
                zoom_range=(1.,1.),
                horizontal_flip=False,
                vertical_flip=False,
                x_interp='constant',
                y_interp='nearest',
                fill_value=0,
                verbose=1):
    
        transform_dict = {
            'rotation_range'    : rotation_range,
            'shift_range'       : shift_range,
            'shear_range'       : shear_range,
            'zoom_range'        : zoom_range,
            'horizontal_flip'   : horizontal_flip,
            'vertical_flip'     : vertical_flip,
            'x_fill_mode'       : x_interp,
            'y_fill_mode'       : y_interp,
            'fill_value'        : fill_value
        }

        super(ImageArraySampler, self).__init__(x=x, 
                                                y=y,
                                                crop_shape=crop_shape,
                                                batch_size=batch_size, 
                                                shuffle=shuffle,
                                                transform_dict=transform_dict,
                                                augment=augment,
                                                sampler_type='image',
                                                verbose=verbose)


class ImageDirectorySampler(BaseDirectorySampler):

    def __init__(self,
                directory,
                input_regex='*',
                target_regex=None,
                batch_size=32, 
                shuffle=True,
                augment=True,
                crop_shape=None,
                rotation_range=0.,
                shift_range=(0.,0.),
                shear_range=0.,
                zoom_range=(1.,1.),
                horizontal_flip=False,
                vertical_flip=False,
                x_interp='constant',
                y_interp='nearest',
                fill_value=0,
                verbose=1):
        
        transform_dict = {
            'rotation_range'    : rotation_range,
            'shift_range'       : shift_range,
            'shear_range'       : shear_range,
            'zoom_range'        : zoom_range,
            'horizontal_flip'   : horizontal_flip,
            'vertical_flip'     : vertical_flip,
            'x_fill_mode'       : x_interp,
            'y_fill_mode'       : y_interp,
            'fill_value'        : fill_value
        }
            
        super(ImageDirectorySampler, self).__init__(directory=directory,
                                                    input_regex=input_regex,
                                                    target_regex=target_regex,
                                                    batch_size=batch_size,
                                                    shuffle=shuffle,
                                                    crop_shape=crop_shape,
                                                    transform_dict=transform_dict,
                                                    augment=augment,
                                                    sampler_type='image',
                                                    verbose=verbose)

class SliceArraySampler(BaseArraySampler):

    def __init__(self, 
                x, 
                y,
                axis=0, 
                batch_size=32, 
                shuffle=True,
                augment=True,
                crop_shape=None,
                rotation_range=0.,
                shift_range=(0.,0.),
                shear_range=0.,
                zoom_range=(1.,1.),
                horizontal_flip=False,
                vertical_flip=False,
                x_interp='constant',
                y_interp='nearest',
                fill_value=0,
                verbose=1):

        transform_dict = {
            'rotation_range'    : rotation_range,
            'shift_range'       : shift_range,
            'shear_range'       : shear_range,
            'zoom_range'        : zoom_range,
            'horizontal_flip'   : horizontal_flip,
            'vertical_flip'     : vertical_flip,
            'x_fill_mode'       : x_interp,
            'y_fill_mode'       : y_interp,
            'fill_value'        : fill_value
        }
        self.axis = axis
        
        super(SliceArraySampler, self).__init__(x=x, 
                                                y=y, 
                                                batch_size=batch_size, 
                                                crop_shape=crop_shape,
                                                shuffle=shuffle,
                                                augment=augment,
                                                transform_dict=transform_dict,
                                                sampler_type='slice',
                                                verbose=verbose)


class SliceDirectorySampler(BaseDirectorySampler):

    def __init__(self, 
                directory,
                config=None, 
                input_regex='*',
                target_regex=None,
                batch_size=32,
                shuffle=False,
                axis=0,
                rotation_range=0.,
                shift_range=(0.,0.),
                shear_range=0.,
                zoom_range=(1.,1.),
                horizontal_flip=False,
                vertical_flip=False,
                x_interp='constant',
                y_interp='nearest',
                fill_value=0.,
                crop_shape=None,
                augment=True,
                verbose=1):
        
        transform_dict = {
            'rotation_range'    : rotation_range,
            'shift_range'       : shift_range,
            'shear_range'       : shear_range,
            'zoom_range'        : zoom_range,
            'horizontal_flip'   : horizontal_flip,
            'vertical_flip'     : vertical_flip,
            'x_fill_mode'       : x_interp,
            'y_fill_mode'       : y_interp,
            'fill_value'        : fill_value
        }
        self.axis = axis
            
        super(SliceDirectorySampler, self).__init__(directory=directory,
                                                    input_regex=input_regex,
                                                    target_regex=target_regex,
                                                    crop_shape=crop_shape,
                                                    batch_size=batch_size,
                                                    shuffle=shuffle,
                                                    transform_dict=transform_dict,
                                                    augment=augment,
                                                    sampler_type='slice',
                                                    verbose=verbose)



