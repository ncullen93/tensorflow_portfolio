
## FUTURE IMPORTS
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

## THIRD PARTY IMPORTS
import nibabel as nib
import scipy.ndimage as ndi
import PIL.Image
import numpy as np
import os
import threading
import fnmatch

def extract_slice(x, y, kwargs):
    slice_x = None
    slice_y = None
    axis = kwargs['axis']
    while True:
        keep_slice  = np.random.randint(0,x.shape[axis])
        if axis == 0:
            slice_x = x[keep_slice,:,:]
            if y is not None:
                slice_y = y[keep_slice,:,:]
        elif axis == 1:
            slice_x = x[:,keep_slice,:]
            if y is not None:
                slice_y = y[:,keep_slice,:]
        elif axis == 2:
            slice_x = x[:,:,keep_slice]
            if y is not None:
                slice_y = y[:,:,keep_slice]

        if y is not None:
            if np.sum(slice_y) > 0:
                break
        else:
            if np.sum(slice_x) > 0:
                break
    return slice_x, slice_y

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix
    
def apply_transform(x, transform_matrix, channel_axis=2, fill_mode='nearest', fill_value=0.):
    x = np.rollaxis(x, channel_axis, 0)
    x = x.astype('float32')
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                final_offset, order=0, mode=fill_mode, cval=fill_value) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def pad_img(img, desired_shape):
    shape_diffs = [int(np.ceil((i_s - d_s))) for d_s,i_s in zip(img.shape,desired_shape)]
    shape_diffs = np.maximum(shape_diffs,0)
    pad_sizes   = [(int(np.ceil(s/2.)),int(np.floor(s/2.))) for s in shape_diffs]
    img = np.pad(img, pad_sizes, mode='constant')
    return img

def needs_padding(img_shape, desired_shape):
    needs_pad = np.sum([s1<s2 for s1,s2 in zip(img_shape,desired_shape)])
    return needs_pad > 0

def random_crop_fn(x, y, kwargs):
    if 'crop_shape' not in kwargs:
        raise Exception('Must give crop_shape to use random cropping')
    if x.ndim == 3:
        raise Exception('Cropping only supported on 2D images')

    crop_height, crop_width = kwargs['crop_shape']
    
    ## pad if necessary
    if needs_padding(x.shape, kwargs['crop_shape']):
        x = pad_img(x, kwargs['crop_shape'])
        if y is not None:
            y = pad_img(y, kwargs['crop_shape'])

    # take random crop 
    h_idx   = np.random.randint(0,x.shape[0]-crop_height+1)
    w_idx   = np.random.randint(0,x.shape[1]-crop_width+1)
    x       = x[h_idx:(h_idx+crop_height),w_idx:(w_idx+crop_width)]
    if y is not None:
        y       = y[h_idx:(h_idx+crop_height),w_idx:(w_idx+crop_width)] 
    return x, y

def random_transform_fn(x, y, kwargs):
    """
    Randomly transform an image from the given parameters

    Transforms:
    - rotate
    - shift
    - shear
    - zoom
    """
    if 'transform_dict' not in kwargs:
        raise Exception('Must give transform_dict to use random transforms')
    T = kwargs['transform_dict']

    # only support tf ordering
    orig_dim = x.ndim
    if y is not None:
        orig_ydim = y.ndim
    if x.ndim == 2:
        x = np.expand_dims(x,-1)
        if y is not None and y.ndim == 2:
            y = np.expand_dims(y,-1)

    img_row_axis = 0
    img_col_axis = 1
    channel_axis = 2

    ### ROTATION
    if T['rotation_range'] > 0:
        theta = np.pi / 180 * np.random.uniform(-T['rotation_range'],
                    T['rotation_range'])
    else:
        theta = 0
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    ### SHIFT HEIGHT
    if T['shift_range'][0] > 0:
        tx = np.random.uniform(-T['shift_range'][0], 
            T['shift_range'][0]) * x.shape[img_row_axis]
    else:
        tx = 0
    ### SHIFT WIDTH
    if T['shift_range'][1] > 0:
        ty = np.random.uniform(-T['shift_range'][1], 
            T['shift_range'][1]) * x.shape[img_col_axis]
    else:
        ty = 0
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    ### SHEAR
    if T['shear_range'] > 0:
        shear = np.random.uniform(-T['shear_range'],T['shear_range'])
    else:
        shear = 0
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    ### ZOOM
    if T['zoom_range'][0] == 1. and T['zoom_range'][1] == 1.:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(T['zoom_range'][0], T['zoom_range'][1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    ### COMBINE MATRICES INTO ONE TRANSFORM MATRIX
    transform_matrix = np.dot(np.dot(np.dot(rotation_matrix,
                                    translation_matrix),
                             shear_matrix),
                      zoom_matrix)
    h, w = x.shape[img_row_axis], x.shape[img_col_axis]
    ### APPLY COMBINED TRANSFORM ON X IMAGE
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis,
                        fill_mode=T['x_fill_mode'], fill_value=T['fill_value'])
    ### APPLY COMBINED TRANSFORM ON Y IMAGE
    if y is not None:
        y = apply_transform(y, transform_matrix, channel_axis,
                    fill_mode=T['y_fill_mode'], fill_value=T['fill_value'])
    ### HORIZONTAL FLIP
    if T['horizontal_flip'] == True:
        if np.random.random() < 0.5:
            x = np.asarray(x).swapaxes(img_col_axis, 0)
            x = x[::-1, ...]
            x = x.swapaxes(0, img_col_axis)
            if y is not None:
                y = np.asarray(y).swapaxes(img_col_axis, 0)
                y = y[::-1, ...]
                y = y.swapaxes(0, img_col_axis)
    ### VERTICAL FLIP
    if T['vertical_flip']:
        if np.random.random() < 0.5:
            x = np.asarray(x).swapaxes(img_row_axis, 0)
            x = x[::-1, ...]
            x = x.swapaxes(0, img_row_axis)
            if y is not None:
                y = np.asarray(y).swapaxes(img_row_axis, 0)
                y = y[::-1, ...]
                y = y.swapaxes(0, img_row_axis)

    if orig_dim == 2:
        x = np.squeeze(x)
        if y is not None and orig_ydim == 2:
            y = np.squeeze(y)
        return x, y
    else:
        return x, y

def load_img(path, grayscale=False, target_size=None):
    img = PIL.Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return img


class BaseArraySampler(object):

    def __init__(self, 
                x, 
                y, 
                batch_size, 
                shuffle,
                crop_shape,
                transform_dict,
                augment,
                sampler_type,
                verbose=1):
        self.verbose=verbose
        self.sampler_type = sampler_type
        self.x = x
        self.nb_samples     = x.shape[0]
        self.input_shape    = x[0].shape
        
        # check if there is a target
        self.y = y
        if self.y is None:
            self.has_target     = False
            self.target_shape   = []
        else:
            self.has_target     = True
            if y.ndim == 1:
                self.target_shape = [1]
            else:
                self.target_shape   = y[0].shape
            assert x.shape[0]==y.shape[0],'X and Y must have same number of samples'

        self.crop_shape = crop_shape
        if crop_shape is not None:
            self.input_shape    = self.crop_shape
            self.target_shape   = self.crop_shape
        
        if type(crop_shape) is int:
            crop_shape = (crop_shape, crop_shape)
        self.crop_shape = crop_shape

        if self.crop_shape is not None:
            self.batch_input_shape  = self.crop_shape
            self.batch_target_shape = self.crop_shape
        else:
            self.batch_input_shape  = self.input_shape
            self.batch_target_shape = self.target_shape

        self.augment        = augment
        self.transform_dict = transform_dict
        if type(self.transform_dict['zoom_range']) is int:
            self.transform_dict['zoom_range']   = (self.transform_dict['zoom_range'],
                self.transform_dict['zoom_range'])
        if type(self.transform_dict['shift_range']) is int:
            self.transform_dict['shift_range']  = (self.transform_dict['shift_range'],
                self.transform_dict['shift_range'])

        self.batch_size = batch_size
        self.shuffle    = shuffle

        # shuffle from the start if given
        if self.shuffle == True:
            perm    = np.random.permutation(self.nb_samples)
            self.x  = self.x[perm]
            self.y  = self.y[perm]

        # INITIALIZE EMPTY PIPELINES
        self.array_kwargs       = {}
        self.x_array_pipeline   = []
        self.y_array_pipeline   = []
        self.xy_array_pipeline  = []
        self.batch_kwargs       = []
        self.x_batch_pipeline   = []
        self.y_batch_pipeline   = []
        self.xy_batch_pipeline  = []

        if self.sampler_type.upper() == 'SLICE':
            self.xy_array_pipeline.append(extract_slice)
            self.array_kwargs['axis'] = self.axis
            # alter batch shapes to reflect slice dims
            self.batch_input_shape = [s for i,s in enumerate(self.input_shape) if i!=self.axis]
            self.batch_target_shape = [s for i,s in enumerate(self.input_shape) if i!=self.axis]
        if self.crop_shape is not None:
            print('Using cropped inputs and targets')
            self.xy_array_pipeline.append(random_crop_fn)
            self.array_kwargs['crop_shape']     = self.crop_shape
        if self.augment == True:
            print('Using data augmentation')
            self.xy_array_pipeline.append(random_transform_fn)
            self.array_kwargs['transform_dict'] = self.transform_dict



        # INDICES FOR SAMPLING ITERATOR
        self.epochs_completed   = 0
        self.index_in_epoch     = 0
        self.start_idx          = 0
        self.batches_seen       = 0

    
    def array_processor(self, x, y):
        for xy_fn in self.xy_array_pipeline:
            x, y    = xy_fn(x, y, self.array_kwargs)
        for x_fn in self.x_array_pipeline:
            x       = x_fn(x, self.array_kwargs)
        for y_fn in self.y_array_pipeline:
            y       = y_fn(y, self.array_kwargs)

        return x, y

    def batch_processor(self, x, y):
        for xy_fn in self.xy_batch_pipeline:
            x, y    = xy_fn(x, y, self.batch_kwargs)
        for x_fn in self.x_batch_pipeline:
            x       = x_fn(x, self.batch_kwargs)
        for y_fn in self.y_batch_pipeline:
            y       = y_fn(y, self.batch_kwargs)

        return x, y

    def next_batch(self):
        # GATHER SAMPLE INDICES FOR CURRENT BATCH
        start_idx           = self.start_idx
        end_idx             = np.minimum(start_idx + self.batch_size, self.nb_samples)
        current_batch_size  = end_idx - start_idx
        self.batches_seen   += 1
        self.start_idx      += self.batch_size # increment start_idx for next batch
        # INITIALIZE EMPTY BATCH
        batch_x     = np.zeros([current_batch_size] + list(self.batch_input_shape))
        if self.has_target:
            batch_y = np.zeros([current_batch_size] + list(self.batch_target_shape))
        # SINGLE ARRAY (SAMPLE) PROCESSING
        for b_idx,s_idx in enumerate(range(start_idx,end_idx)):
            raw_x = self.x[s_idx]
            if self.has_target:
                raw_y           = self.y[s_idx]
                arr_x, arr_y    = self.array_processor(raw_x, raw_y)
                batch_x[b_idx], batch_y[b_idx]  = arr_x, arr_y
            else:
                arr_x           = self.array_processor(raw_x, None)
                batch_x[b_idx]  = arr_x

        # ENTIRE BATCH PROCESSING
        if self.has_target:
            batch_x, batch_y    = self.batch_processor(batch_x, batch_y)
            batch_return        = (batch_x,batch_y)
        else:
            batch_x, _          = self.batch_processor(batch_x, None)
            batch_return        = batch_x

        # INDEX HOUSEKEEPING ON FINAL BATCH IN THE EPOCH
        if end_idx >= self.nb_samples:
            self.epochs_completed  += 1
            self.start_idx          = 0 # reset start_idx
            # Shuffle the data for next epoch if necessary
            if self.shuffle == True:
                perm    = np.random.permutation(self.nb_samples)
                self.x  = self.x[perm]
                self.y  = self.y[perm]

        return batch_return

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next_batch(*args, **kwargs)

    def next(self, *args, **kwargs):
        return self.next_batch(*args, **kwargs)
                

class BaseDirectorySampler(object):

    def __init__(self, 
                directory,
                input_regex,
                target_regex,
                batch_size, 
                shuffle,
                crop_shape,
                augment,
                transform_dict,
                sampler_type,
                verbose=1):
        self.verbose        = verbose
        self.sampler_type   = sampler_type
        self.directory      = directory
        if not self.directory.endswith('/'):
            self.directory += '/'

        self.input_regex    = input_regex
        self.batch_size     = batch_size
        # HANDLE TARGET TYPE
        self.target_regex   = target_regex
        if self.target_regex is None:
            self.has_target = False
        else:
            self.has_target = True

        self.shuffle        = shuffle
        self.transform_dict = transform_dict 
        if type(self.transform_dict['zoom_range']) is int:
            self.transform_dict['zoom_range']   = (self.transform_dict['zoom_range'],
                self.transform_dict['zoom_range'])
        if type(self.transform_dict['shift_range']) is int:
            self.transform_dict['shift_range']  = (self.transform_dict['shift_range'],
                self.transform_dict['shift_range'])

        if type(crop_shape) is int:
            crop_shape  = (crop_shape, crop_shape)
        self.crop_shape = crop_shape
        self.augment    = augment


        # INITIALIZE CACHE
        #self.max_cache     = 0 # pass max_cache as argument
        #self.has_cache     = False
        #if self.max_cache is not None:
        #   self.has_cache          = True
        #   self.cache_check        = [False] * self.max_cache
        #   self.target_cache_check = [False] * self.max_cache

        # GATHER SUB-DIRECTORIES IN THE MAIN DIRECTORY
        subdirs = []
        for subdir in sorted(os.listdir(self.directory)):
            if os.path.isdir(os.path.join(self.directory, subdir)):
                subdirs.append(subdir)
        self.nb_subdirs         = len(subdirs)
        self.subdir_indices     = dict(zip(subdirs, np.arange(self.nb_subdirs)))

        # WALK THROUGH EACH SUB-DIRECTORY TO GATHER FILENAMES
        read_formats = {'.png','.jpg','.jpeg','.bmp','.npy', '.nii.gz'}
        self.nb_samples         = 0
        self.input_filenames    = []
        self.input_classes      = []
        if self.has_target:
            self.nb_target_samples  = 0
            self.target_filenames   = []
            self.target_classes     = []
        for subdir_idx, subdir in enumerate(subdirs):
            subdir_path = os.path.join(self.directory, subdir)
            for root, subdir, filenames in os.walk(subdir_path):
                for fname in filenames:
                    is_valid = False
                    for extension in read_formats:
                        if fname.lower().endswith(extension):
                            is_valid =True
                    if is_valid:
                        rel_path = root.split(self.directory)[1]
                        # IF FILE MATCHES INPUT REGEX
                        if fnmatch.fnmatch(fname, self.input_regex):
                            self.input_filenames.append(os.path.join(rel_path,fname))
                            self.input_classes.append(subdir_idx)
                            self.nb_samples += 1
                        # IF FILE MATCHES TARGET REFEX
                        elif self.has_target and fnmatch.fnmatch(fname, self.target_regex):
                            self.target_filenames.append(os.path.join(rel_path,fname))
                            self.target_classes.append(subdir_idx)  
                            self.nb_target_samples += 1         
        
        # RAISE ERROR IF NO FILES FOUND
        assert len(self.input_filenames)>0, 'No valid file is found in the target directory.'

        # INDICES FOR SAMPLING ITERATOR
        self.epochs_completed   = 0
        self.index_in_epoch     = 0
        self.start_idx          = 0
        self.batches_seen       = 0

        ## INITIALIZE EMPTY PIPELINES
        # file pipelines
        self.file_kwargs        = {}
        self.x_file_pipeline    = []
        self.y_file_pipeline    = []
        self.xy_file_pipeline   = []

        # array pipelines
        self.array_kwargs       = {}
        self.x_array_pipeline   = []
        self.y_array_pipeline   = []
        self.xy_array_pipeline  = []
        if self.sampler_type.upper() == 'SLICE':
            self.xy_array_pipeline.append(extract_slice)
            self.array_kwargs['axis'] = self.axis
        if self.crop_shape is not None:
            print('Using cropped inputs and targets')
            self.xy_array_pipeline.append(random_crop_fn)
            self.array_kwargs['crop_shape'] = self.crop_shape
        if self.augment == True:
            print('Using data augmentation')
            self.xy_array_pipeline.append(random_transform_fn)
            self.array_kwargs['transform_dict'] = self.transform_dict
            
        # batch pipelines
        self.batch_kwargs       = {}
        self.x_batch_pipeline   = []
        self.y_batch_pipeline   = []
        self.xy_batch_pipeline  = []

        if not self.has_target:
            if self.verbose > 0:
                print('Found %d input images for %d main classes' % (self.nb_samples, self.nb_subdirs))
        else:
            if self.verbose > 0:
                print('Found %d input images and %d target images' %(self.nb_samples,self.nb_target_samples))

        # INFER INPUT SHAPE (AND TARGET SHAPE) IF NOT GIVEN
        if not self.has_target:
            in_fname                    = os.path.join(self.directory,self.input_filenames[0])
            in_array,_                  = self.file_processor(in_fname,None)
            self.input_shape            = in_array.shape
            in_pr_array,_               = self.array_processor(in_array,None)
            self.batch_input_shape      = in_pr_array.shape
            if self.crop_shape is None:
                if self.verbose > 0:
                    print('Inferred Input Shape: ' ,self.input_shape)
            else:
                if self.verbose > 0:
                    print('Inferred Input Shape: ', self.input_shape , ' -> ' , self.batch_input_shape)
        else:
            in_fname                    = os.path.join(self.directory,self.input_filenames[0])
            tar_fname                   = os.path.join(self.directory, self.target_filenames[0])
            in_array, tar_array         = self.file_processor(in_fname, tar_fname)
            self.input_shape            = in_array.shape
            self.target_shape           = tar_array.shape
            in_pr_array, tar_pr_array   = self.array_processor(in_array, tar_array)
            self.batch_input_shape      = in_pr_array.shape
            self.batch_target_shape     = tar_pr_array.shape
            if self.crop_shape is None:
                if self.verbose > 0:
                    print('Inferred Input Shape: ' ,self.batch_input_shape)
                    print('Inferred Target Shape: ' ,self.batch_target_shape)
            else:
                if self.verbose > 0:
                    print('Inferred Input Shape: ' ,self.input_shape , ' -> ' , self.batch_input_shape)
                    print('Inferred Target Shape: ' ,self.target_shape , ' -> ' , self.batch_target_shape)

        self.lock = threading.Lock()


    def file_processor(self, x, y):
        for xy_fn in self.xy_file_pipeline:
            x, y    = xy_fn(x, y, self.file_kwargs)
        for x_fn in self.x_file_pipeline:
            x       = x_fn(x, self.file_kwargs)
        for y_fn in self.y_file_pipeline:
            y       = y_fn(y, self.file_kwargs)

        if x.endswith('npy'):
            x = np.load(x)
        elif x.endswith('nii.gz'):
            x = nib.load(x).get_data()

        if y is not None:
            if y.endswith('npy'):
                y = np.load(y)
            elif y.endswith('nii.gz'):
                y = nib.load(y).get_data()
            else:
                y = load_img(y)

        return x, y

    def array_processor(self, x, y):
        for xy_fn in self.xy_array_pipeline:
            x, y    = xy_fn(x, y, self.array_kwargs)
        for x_fn in self.x_array_pipeline:
            x       = x_fn(x, self.array_kwargs)
        for y_fn in self.y_array_pipeline:
            y       = y_fn(y, self.array_kwargs)

        return x, y

    def batch_processor(self, x, y):
        for xy_fn in self.xy_batch_pipeline:
            x, y    = xy_fn(x, y, self.batch_kwargs)
        for x_fn in self.x_batch_pipeline:
            x       = x_fn(x, self.batch_kwargs)
        for y_fn in self.y_batch_pipeline:
            y       = y_fn(y, self.batch_kwargs)

        return x, y

    def next_batch(self):
        """
        Sample next batch 
        """
        # GATHER SAMPLE INDICES FOR CURRENT BATCH
        start_idx           = self.start_idx
        end_idx             = np.minimum(start_idx + self.batch_size, self.nb_samples)
        current_batch_size  = end_idx - start_idx
        self.batches_seen   += 1
        self.start_idx      += self.batch_size # increment start_idx for next batch

        # INITIALIZE EMPTY BATCH
        batch_x     = np.zeros([current_batch_size] + list(self.batch_input_shape))
        if self.has_target:
            batch_y = np.zeros([current_batch_size] + list(self.batch_target_shape))
        
        # SINGLE FILE & ARRAY (SAMPLE) PROCESSING
        for b_idx,s_idx in enumerate(range(start_idx,end_idx)):
            file_x = os.path.join(self.directory,self.input_filenames[s_idx])
            if self.has_target:
                file_y = os.path.join(self.directory,self.target_filenames[s_idx])
                raw_x, raw_y    = self.file_processor(file_x, file_y)
                arr_x, arr_y    = self.array_processor(raw_x, raw_y)
                batch_x[b_idx],batch_y[b_idx]   = arr_x, arr_y
            else:
                raw_x,_         = self.file_processor(file_x, None)
                arr_x,_         = self.array_processor(raw_x, None)             
                batch_x[b_idx]  = arr_x

        # ENTIRE BATCH PROCESSING
        if self.has_target:
            batch_x, batch_y    = self.batch_processor(batch_x, batch_y)
            batch_return        = (batch_x,batch_y)
        else:
            batch_x, _          = self.batch_processor(batch_x, None)
            batch_return        = batch_x

        # INDEX HOUSEKEEPING ON FINAL BATCH IN THE EPOCH
        if end_idx >= self.nb_samples:
            self.epochs_completed  += 1
            self.start_idx          = 0 # reset start_idx
            # Shuffle the data for next epoch if necessary
            if self.shuffle == True:
                perm                        = np.random.permutation(self.nb_samples)
                self.input_filenames        = [self.input_filenames[i] for i in perm]
                self.input_classes          = [self.input_classes[i] for i in perm]
                if self.has_target:
                    self.target_filenames   = [self.target_filenames[i] for i in perm]
                    self.target_classes     = [self.target_classes[i] for i in perm]

        return batch_return
        
    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        with self.lock:
            return self.next_batch(*args, **kwargs)

    def next(self, *args, **kwargs):
        with self.lock:
            return self.next_batch(*args, **kwargs)

