import sys
from numbers import Number

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Mapping, Union, Callable

from muse.utils.general_utils import is_array
from attrdict import AttrDict

numpy_to_torch_dtype_dict = {
    bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128
}

reduce_map_fn = {
    'sum': lambda t: t.sum(),
    'mean': lambda t: t.mean(),
    'max': lambda t: t.max(),
    'min': lambda t: t.min(),
}

torch_to_numpy_dtype_dict = {val: key for key, val in numpy_to_torch_dtype_dict.items()}

torch_mappable = lambda dt: dt in numpy_to_torch_dtype_dict.keys() or dt in torch_to_numpy_dtype_dict.keys() \
                            or (isinstance(dt, np.dtype) and dt.type in numpy_to_torch_dtype_dict.keys())


def dc_torch_mappable(dc):
    return dc.leaf_filter(lambda k, v: torch_mappable(v.dtype))


class ShapedModule(object):
    """
    Interface to allow shape checking for a module.
    enables output shape checking on forward() call, meant to be used with torch.nn.Module

    TODO add this as a parent class to things
    """

    def output_shape(self, input_shape: torch.Size) -> torch.Size:
        return input_shape


############################
# Array shape modification #
############################

def get_zeroth_horizon(x):
    if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        return x[:, 0]
    else:
        return x


def add_horizon_dim(x):
    if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        return x[:, None]
    else:
        return x


def unsqueeze_n(x, n, dim=0):
    for _ in range(n):
        x = x.unsqueeze(dim)
    return x


# horizon on AttrDict
def dc_add_horizon_dim(dc):
    return dc.leaf_apply(lambda arr: arr[:, None])


# batch on AttrDict
def dc_add_batch_dim(dc):
    return dc.leaf_apply(lambda arr: arr[None])


def to_scalar(arr):
    if isinstance(arr, Number):
        return arr
    elif is_array(arr):
        return arr.mean().item()
    else:
        raise NotImplementedError(f"Non-scalar: {type(arr)}")


def expand_at_single(x, size, dim):
    """
    Expand a tensor at a single dimension @dim by @size
    Args:
        x (torch.Tensor): input tensor
        size (int): size to expand
        dim (int): dimension to expand
    Returns:
        y (torch.Tensor): expanded tensor
    """
    assert dim < x.ndimension()
    assert x.shape[dim] == 1
    expand_dims = [-1] * x.ndimension()
    expand_dims[dim] = size
    return x.expand(*expand_dims)


def expand_at(x, size, dim):
    """
    Expand all tensors in nested dictionary or list or tuple at a single
    dimension @dim by @size.
    Args:
        x (AttrDict)
        size (int): size to expand
        dim (int): dimension to expand
    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    if isinstance(x, AttrDict):
        return x.leaf_apply(lambda arr: expand_at_single(arr, size, dim))
    else:
        return expand_at_single(x, size, dim)


def unsqueeze_expand_at(x, size, dim):
    """
    Unsqueeze and expand a tensor at a dimension @dim by @size.
    Args:
        x (AttrDict or tensor)
        size (int): size to expand
        dim (int): dimension to unsqueeze and expand
    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    if isinstance(x, AttrDict):
        x = x.leaf_apply(lambda arr: arr.unsqueeze(dim))
    else:
        x = x.unsqueeze(dim)
    return expand_at(x, size, dim)


def cat_any(vs, *args, dim=0, **kwargs):
    if isinstance(vs[0], np.ndarray):
        return np.concatenate(vs, *args, axis=dim, **kwargs)
    else:
        return torch.cat(vs, *args, dim=dim, **kwargs)


def concatenate(input_dict: AttrDict, names, dim=0, default_shape=None):
    # print(names, list(input_dict.leaf_keys()))
    all = []
    for n in names:
        all.append(input_dict[n])
    if len(all) > 0:
        if type(all[0]) is torch.Tensor:
            return torch.cat(all, dim=dim)
        else:
            return np.concatenate(all, axis=dim)
    elif default_shape is not None:
        return np.empty(list(default_shape) + [0], dtype=np.float32)
    else:
        raise Exception(
            "No elements to concatenate for names: %s from keys: %s" % (names, list(input_dict.leaf_keys())))


def split_dim(torch_in, dim, new_shape):
    sh = list(torch_in.shape)
    if dim < 0:
        dim = len(sh) + dim
    assert dim < len(sh)
    assert sh[dim] == np.prod(new_shape), [sh[dim], new_shape, dim]
    new_shape = sh[:dim] + list(new_shape) + sh[dim + 1:]
    return torch_in.view(new_shape)


def split_dim_np(np_in, axis, new_shape):
    sh = list(np_in.shape)
    if axis < 0:
        axis = len(sh) + axis
    assert axis < len(sh)
    assert sh[axis] == np.prod(new_shape)
    new_shape = sh[:axis] + list(new_shape) + sh[axis + 1:]
    return np_in.reshape(new_shape)


def unsqueeze_then_gather(arr, idxs, dim):
    # idxs is (N1 .. Nj)
    # arr  is (N1 .. Nj M Nj+1 ... Nn)
    dim = dim % len(arr.shape)
    # this will unsqueeze idxs to match (N1 .. Nj Nj+1 ... Nn), and then gather
    assert list(arr.shape[:dim]) == list(
        idxs.shape), f"Indices must have same pre shape as arr: {idxs.shape}, {arr.shape}, dim={dim}"
    idxs = split_dim(idxs[..., None], dim=-1, new_shape=[1] * (len(arr.shape) - dim))
    new_shape = list(arr.shape[:dim]) + [1] + list(arr.shape[dim + 1:])
    idxs = torch.broadcast_to(idxs, new_shape)
    gathered = torch.gather(arr, dim=dim, index=idxs)
    return gathered.squeeze(dim)


def combine_dims(torch_in, start_dim, num_dims=2):
    sh = list(torch_in.shape)
    if start_dim < 0:
        start_dim = len(sh) + start_dim
    assert start_dim < start_dim + num_dims <= len(sh)
    new_sh = sh[:start_dim] + [-1] + sh[start_dim + num_dims:]
    return torch_in.view(new_sh)


def combine_dims_np(np_in, start_axis, num_axes=2):
    sh = list(np_in.shape)
    if start_axis < 0:
        start_axis = len(sh) + start_axis
    assert start_axis < start_axis + num_axes <= len(sh)
    new_shape = sh[:start_axis] + [-1] + sh[start_axis + num_axes:]
    if np.prod(sh) > 0:
        return np_in.reshape(new_shape)
    else:
        comb_shape = np.prod(sh[start_axis:start_axis + num_axes])
        new_shape[start_axis] = int(comb_shape)
        return np.zeros_like(np_in, shape=new_shape)


def combine_after_dim(arr, start_dim, allow_no_dim=False):
    max = len(arr.shape)
    if start_dim < 0:
        start_dim = max + start_dim
    if start_dim == max - 1:
        # already combined to this level
        return arr
    elif allow_no_dim and start_dim == max:
        return arr[..., None]  # add on a final dim
    elif isinstance(arr, torch.Tensor):
        return combine_dims(arr, start_dim, max - start_dim)
    else:
        return combine_dims_np(arr, start_dim, max - start_dim)


def combine_after_last_dim(inputs: AttrDict, max_dim=np.inf):
    assert max_dim > 0
    # will be min(min(len(arr.shape) for all arrs), seed)
    min_len = inputs.leaf_reduce(lambda red, val: min(red, len(val.shape) if is_array(val) else np.inf),
                                 seed=max_dim + 1)
    if 0 < min_len < np.inf:
        return inputs.leaf_apply(lambda arr: combine_after_dim(arr, int(min_len) - 1))
    else:
        return inputs


def broadcast_dims_np(arr: np.ndarray, axes: List[int], new_shape: List[int]):
    assert len(axes) == len(new_shape), [axes, new_shape]
    sh = list(arr.shape)
    axes = [i % len(sh) for i in axes]
    new_sh = sh.copy()
    for i, ax in enumerate(axes):
        new_sh[ax] = new_shape[i]
    return np.broadcast_to(arr, new_sh)


def broadcast_dims(arr: torch.Tensor, dims: List[int], new_shape: List[int]):
    assert len(dims) == len(new_shape), [dims, new_shape]
    sh = [-1] * len(arr.shape)
    for i, ax in enumerate(dims):
        sh[ax] = new_shape[i]
    return arr.expand(sh)


def pad_dims(arr: Union[np.ndarray, torch.Tensor], dims: List[int], new_dims: List[int], val=0., mode='constant',
             after=True, delta=False):
    assert len(dims) == len(new_dims) > 0, [len(dims), len(new_dims), dims, new_dims]
    if not delta:
        # pad space check
        assert all(arr.shape[dim] <= desired for dim, desired in zip(dims, new_dims))
        # subtract each entry to get delta
        new_dims = [desired - arr.shape[dim] for dim, desired in zip(dims, new_dims)]

    # ((before0, after(right)0), ... beforeN, afterN)
    pads = [[0, 0] for _ in range(len(arr.shape))]
    b = int(after)  # pad right if after = True
    for dim, desired in zip(dims, new_dims):
        pads[dim][b] = desired

    if isinstance(arr, torch.Tensor):
        tpad = []
        for p in pads[::-1]:
            tpad += p
        return F.pad(arr, tpad, mode=mode, value=val)
    else:
        return np.pad(arr, pads, mode=mode, constant_values=val)


def concat_apply_split(input_dict: AttrDict, names, func, combine_dim=-1):
    relevant = input_dict > names
    names_to_shapes = relevant.leaf_apply(lambda arr: list(arr.shape))
    arr = concatenate(relevant.leaf_apply(lambda arr: combine_after_dim(arr, combine_dim)), names, dim=combine_dim)
    new_arr = func(arr)
    return unconcatenate(new_arr, names, names_to_shapes)


def combine_then_concatenate(input_dict, names, dim=-1):
    # will combine after dim, then concatenate along that dim.
    relevant = input_dict > names
    return concatenate(relevant.leaf_apply(lambda arr: combine_after_dim(arr, dim)), names, dim=dim)


def unconcatenate(x, names, names_to_shapes, dtypes=None, outs=None, copy=False):
    """

    :param x: inputs flat combined array
    :param names:
    :param names_to_shapes:
    :param dtypes: if specified, outs will by copied to the given dtypes. Only works if outs is None or copy = False
    :param outs: if copy = True, writes to outs from x
                if copy = False, fills outs (AttrDict) but does not copy to existing out arrays.
    :param copy:
    :return:
    """

    assert not (dtypes is not None and outs is not None and copy), "Cannot specify dtypes AND outs when copy=True"

    d = AttrDict()  # where we write to initially
    prepend_shape = list(x.shape[:-1])
    shapes = [list(names_to_shapes[n]) for n in names]
    flat_shapes = [int(np.prod(sh)) for sh in shapes]
    # with timeit("split"):
    if isinstance(x, np.ndarray):
        chunked = np.split(x, np.cumsum(flat_shapes)[:-1], -1)
    else:
        chunked = torch.split(x, flat_shapes, -1)
    # with timeit("copyto"):
    for i, n in enumerate(names):
        # with timeit(f"copy_to_{n}"):
        d[n] = chunked[i].reshape(prepend_shape + shapes[i])

        if dtypes is not None:
            d[n] = d[n].astype(dtypes[n], copy=copy) if isinstance(x, np.ndarray) else d[n].to(dtype=dtypes[n])

        # mem copy if outs is specified, or just transfer.
        if copy and outs is not None:
            dst = outs[n]
            src = d[n]
            # print(n, dst.shape, dst.dtype)
            np.copyto(dst, src, casting='unsafe')
            d[n] = outs[n]  # outs will be correct.
        elif outs is not None:
            outs[n] = d[n]

    return d


def view_unconcatenate(x, names, names_to_shapes):
    """
    Gets views of the big x array for each name, sliced and "viewed" or reshaped (memory efficient)

    :param x: inputs flat combined array
    :param names:
    :param names_to_shapes:
    :return:
    """

    d = AttrDict()  # where we write to initially
    prepend_shape = list(x.shape[:-1])

    idx_start = 0

    for name in names:
        shape = list(names_to_shapes[name])
        flat_shape = int(np.prod(shape))
        d[name] = x[..., idx_start:idx_start + flat_shape]
        if isinstance(d[name], np.ndarray):
            d[name] = d[name].reshape(prepend_shape + shape)
        else:
            d[name] = d[name].view(prepend_shape + shape)

        idx_start += flat_shape  # increment for next key

    assert idx_start == x.shape[-1], [idx_start, x.shape[-1], names]

    return d


def get_indices_for_flat(indices_names, names, names_to_shapes):
    """
    Gets the indices that would get "indices_names" from a concatenated flat array with "names".

    Names should be in order, but "indices_names" might not be! we go with the order in names.
        therefore the mapping from indices -> out_names will be permuted (not in indices_names order)

    :param x: inputs flat combined array
    :param indices_names:
    :param names:
    :param names_to_shapes:
    :return:
    """
    idx_count = 0
    indices = []
    for name in names:
        shape = list(names_to_shapes[name])
        flat_shape = int(np.prod(shape))
        if name in indices_names:
            indices.extend(range(idx_count, idx_count + flat_shape))
        idx_count += flat_shape

    return np.asarray(indices)


############################
#     BASIC conversion     #
############################

def to_torch(numpy_in, device="cuda", check=False):
    if check and isinstance(numpy_in, torch.Tensor):
        return numpy_in.to(device)
    if check and not isinstance(numpy_in, np.ndarray):
        return numpy_in
    else:
        return torch.from_numpy(numpy_in.copy()).to(device)


def to_numpy(torch_in, check=False):
    if check and isinstance(torch_in, np.ndarray):
        return torch_in
    return torch_in.detach().cpu().numpy()


def torch_clip(torch_in, low, high):
    clip_low = torch.where(torch_in >= low, torch_in, low)
    return torch.where(clip_low <= high, clip_low, high)


def torch_clip_norm(arr, norm, dim=None):
    scale = norm / (torch.norm(arr, dim=dim) + 1e-11)
    return arr * torch.minimum(scale, torch.tensor([1], dtype=scale.dtype))


def numel(arr: Union[np.ndarray, torch.Tensor]):
    if isinstance(arr, np.ndarray):
        return arr.size
    elif is_array(arr):
        return arr.numel()


## others


def disable_gradients(model: torch.nn.Module):
    ls_of_prev = []
    for param in model.parameters():
        ls_of_prev.append(param.requires_grad)
        param.requires_grad = False

    return ls_of_prev


def enable_gradients(model: torch.nn.Module, which_params=None):
    if which_params is None:
        which_params = [True] * len(list(model.parameters()))

    for param, prev_value in zip(model.parameters(), which_params):
        param.requires_grad = prev_value


class torch_disable_grad:
    """
    Context manager that will set all parameters of a Module to requires_grad=False.
    This disables logging gradients for this module while << still enabling gradient tracking >>
    """

    def __init__(self, model: torch.nn.Module, eval_mode=True):
        self._model = model
        self._ls_of_prev = None
        self._eval = eval_mode
        self._pre_mode = None

    def __enter__(self):
        self._ls_of_prev = disable_gradients(self._model)
        if self._eval:
            self._pre_mode = self._model.training
            self._model.eval()

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self._ls_of_prev is not None, "bug"
        enable_gradients(self._model, self._ls_of_prev)
        if self._eval:
            assert self._pre_mode is not None, "bug"
            self._model.train(self._pre_mode)


# Concat-able AttrDict. Frozen key map after init. Values can be changed
class CAttrDict(AttrDict):

    def __init__(self, names, after_dim=-1):
        super(CAttrDict, self).__init__()
        self.__dict__['_fixed_names'] = list(names)
        self.__dict__['_after_dim'] = after_dim  # after which dim to combine and concatenate

        for n in names:
            self[n] = None
        self.freeze()  # now it cannot be changed and the order is fixed

        self.__dict__['_curr_dim'] = self._after_dim
        self.__dict__['_concat_arr'] = None

    def __setitem__(self, key, value):
        # Only top node will be a CAttrDict.
        assert key in self.__dict__["_fixed_names"], f"Cannot add new key {key}"

        if isinstance(key, str) and '/' in key:
            key_split = key.split('/')
            curr_key = key_split[0]
            next_key = '/'.join(key_split[1:])
            if not self.has_key(curr_key):
                new_d = AttrDict()
                new_d[next_key] = value
                super(AttrDict, self).__setitem__(curr_key, new_d)
            else:
                self[curr_key][next_key] = value
        else:
            super(AttrDict, self).__setitem__(key, value)

        self.__dict__["_concat_arr"] = None  # invalidates concatenation cache

    @staticmethod
    def from_dynamic(input_dict: AttrDict, order=None, concat_arr=None, after_dim=-1):
        order = order or input_dict.list_leaf_keys()
        d = CAttrDict(order)
        for n in order:
            d[n] = input_dict[n]  # must be present
            assert is_array(d[n]), [n, type(d[n])]

        # initialize concatenation state
        d.__dict__["_concat_arr"] = concat_arr
        d.__dict__["_curr_dim"] = after_dim

        return d

    def concat(self, dim=None):
        if self.__dict__["_concat_arr"] is None or (dim is not None and dim != self.__dict__["_curr_dim"]):
            dim = dim if dim is not None else self.__dict__["_after_dim"]
            self.__dict__["_concat_arr"] = combine_then_concatenate(self, self.__dict__["_fixed_names"], dim=dim)

        return self.__dict__["_concat_arr"]


def get_augment_fn(std):
    std_arr = to_torch(np.asarray(std)[None], device="cpu")

    def fn(arr, **kwargs):
        nonlocal std_arr
        if std_arr.device != arr.device:
            std_arr = std_arr.to(device=arr.device)
        return arr + std_arr * torch.randn_like(arr)

    return fn


def get_masked_augment_fn(std, mask_key='mask', prefix='read_inputs', corr=False):
    def fn(arr, memory=None, **kwargs):
        mask = memory >> f"{prefix}/{mask_key}"
        mask = unsqueeze_n(mask, len(arr.shape) - len(mask.shape), dim=-1)
        if corr:
            return arr + mask * std * torch.randn_like(arr[:, :1])
        else:
            return arr + mask * std * torch.randn_like(arr)

    return fn


def randint_between(t1, t2, fn=lambda x: x):
    t1, t2 = torch.broadcast_tensors(t1, t2)
    assert torch.all(t1 < t2)
    delta = (t2 - t1).to(dtype=torch.int)
    # [0, b-a)
    eps = delta * fn(torch.rand_like(t1, dtype=torch.float))
    # truncate
    eps.trunc_()
    return t1 + eps.to(dtype=torch.int)


def uniform(t1, t2):
    pass


def crop_image_from_indices(images, crop_indices, crop_height, crop_width):
    """
    https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/utils/obs_utils.py
    
    Crops images at the locations specified by @crop_indices. Crops will be 
    taken across all channels.
    Args:
        images (torch.Tensor): batch of images of shape [..., C, H, W]
        crop_indices (torch.Tensor): batch of indices of shape [..., N, 2] where
            N is the number of crops to take per image and each entry corresponds
            to the pixel height and width of where to take the crop. Note that
            the indices can also be of shape [..., 2] if only 1 crop should
            be taken per image. Leading dimensions must be consistent with
            @images argument. Each index specifies the top left of the crop.
            Values must be in range [0, H - CH - 1] x [0, W - CW - 1] where
            H and W are the height and width of @images and CH and CW are
            @crop_height and @crop_width.
        crop_height (int): height of crop to take
        crop_width (int): width of crop to take
    Returns:
        crops (torch.Tesnor): cropped images of shape [..., C, @crop_height, @crop_width]
    """

    # make sure length of input shapes is consistent
    assert crop_indices.shape[-1] == 2
    ndim_im_shape = len(images.shape)
    ndim_indices_shape = len(crop_indices.shape)
    assert (ndim_im_shape == ndim_indices_shape + 1) or (ndim_im_shape == ndim_indices_shape + 2)

    # maybe pad so that @crop_indices is shape [..., N, 2]
    is_padded = False
    if ndim_im_shape == ndim_indices_shape + 2:
        crop_indices = crop_indices.unsqueeze(-2)
        is_padded = True

    # make sure leading dimensions between images and indices are consistent
    assert images.shape[:-3] == crop_indices.shape[:-2]

    device = images.device
    image_c, image_h, image_w = images.shape[-3:]
    num_crops = crop_indices.shape[-2]

    # make sure @crop_indices are in valid range
    assert (crop_indices[..., 0] >= 0).all().item()
    assert (crop_indices[..., 0] < (image_h - crop_height)).all().item()
    assert (crop_indices[..., 1] >= 0).all().item()
    assert (crop_indices[..., 1] < (image_w - crop_width)).all().item()

    # convert each crop index (ch, cw) into a list of pixel indices that correspond to the entire window.

    # 2D index array with columns [0, 1, ..., CH - 1] and shape [CH, CW]
    crop_ind_grid_h = torch.arange(crop_height).to(device)
    crop_ind_grid_h = unsqueeze_expand_at(crop_ind_grid_h, size=crop_width, dim=-1)
    # 2D index array with rows [0, 1, ..., CW - 1] and shape [CH, CW]
    crop_ind_grid_w = torch.arange(crop_width).to(device)
    crop_ind_grid_w = unsqueeze_expand_at(crop_ind_grid_w, size=crop_height, dim=0)
    # combine into shape [CH, CW, 2]
    crop_in_grid = torch.cat((crop_ind_grid_h.unsqueeze(-1), crop_ind_grid_w.unsqueeze(-1)), dim=-1)

    # Add above grid with the offset index of each sampled crop to get 2d indices for each crop.
    # After broadcasting, this will be shape [..., N, CH, CW, 2] and each crop has a [CH, CW, 2]
    # shape array that tells us which pixels from the corresponding source image to grab.
    grid_reshape = [1] * len(crop_indices.shape[:-1]) + [crop_height, crop_width, 2]
    all_crop_inds = crop_indices.unsqueeze(-2).unsqueeze(-2) + crop_in_grid.reshape(grid_reshape)

    # For using @torch.gather, convert to flat indices from 2D indices, and also
    # repeat across the channel dimension. To get flat index of each pixel to grab for 
    # each sampled crop, we just use the mapping: ind = h_ind * @image_w + w_ind
    all_crop_inds = all_crop_inds[..., 0] * image_w + all_crop_inds[..., 1]  # shape [..., N, CH, CW]
    all_crop_inds = unsqueeze_expand_at(all_crop_inds, size=image_c, dim=-3)  # shape [..., N, C, CH, CW]
    all_crop_inds = combine_after_dim(all_crop_inds, start_dim=-2)  # shape [..., N, C, CH * CW]

    # Repeat and flatten the source images -> [..., N, C, H * W] and then use gather to index with crop pixel inds
    images_to_crop = unsqueeze_expand_at(images, size=num_crops, dim=-4)
    images_to_crop = combine_after_dim(images_to_crop, start_dim=-2)
    crops = torch.gather(images_to_crop, dim=-1, index=all_crop_inds)
    # [..., N, C, CH * CW] -> [..., N, C, CH, CW]
    reshape_axis = len(crops.shape) - 1
    crops = split_dim(crops, reshape_axis, [crop_height, crop_width])

    if is_padded:
        # undo padding -> [..., C, CH, CW]
        crops = crops.squeeze(-4)
    return crops


def sample_random_image_crops(images, crop_height, crop_width, num_crops, pos_enc=False):
    """
    https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/utils/obs_utils.py
    
    For each image, randomly sample @num_crops crops of size (@crop_height, @crop_width), from
    @images.
    Args:
        images (torch.Tensor): batch of images of shape [..., C, H, W]
        crop_height (int): height of crop to take
        
        crop_width (int): width of crop to take
        num_crops (n): number of crops to sample
        pos_enc (bool): if True, also add 2 channels to the outputs that gives a spatial 
            encoding of the original source pixel locations. This means that the
            output crops will contain information about where in the source image 
            it was sampled from.
    Returns:
        crops (torch.Tensor): crops of shape (..., @num_crops, C, @crop_height, @crop_width) 
            if @pos_enc is False, otherwise (..., @num_crops, C + 2, @crop_height, @crop_width)
        crop_inds (torch.Tensor): sampled crop indices of shape (..., N, 2)
    """
    device = images.device

    # maybe add 2 channels of spatial encoding to the source image
    source_im = images
    if pos_enc:
        # spatial encoding [y, x] in [0, 1]
        h, w = source_im.shape[-2:]
        pos_y, pos_x = torch.meshgrid(torch.arange(h), torch.arange(w))
        pos_y = pos_y.float().to(device) / float(h)
        pos_x = pos_x.float().to(device) / float(w)
        position_enc = torch.stack((pos_y, pos_x))  # shape [C, H, W]

        # unsqueeze and expand to match leading dimensions -> shape [..., C, H, W]
        leading_shape = source_im.shape[:-3]
        position_enc = position_enc[(None,) * len(leading_shape)]
        position_enc = position_enc.expand(*leading_shape, -1, -1, -1)

        # concat across channel dimension with input
        source_im = torch.cat((source_im, position_enc), dim=-3)

    # make sure sample boundaries ensure crops are fully within the images
    _, image_h, image_w = source_im.shape[-3:]
    max_sample_h = image_h - crop_height
    max_sample_w = image_w - crop_width

    # Sample crop locations for all tensor dimensions up to the last 3, which are [C, H, W].
    # Each gets @num_crops samples - typically this will just be the batch dimension (B), so 
    # we will sample [B, N] indices, but this supports having more than one leading dimension,
    # or possibly no leading dimension.
    #
    # Trick: sample in [0, 1) with rand, then re-scale to [0, M) and convert to long to get sampled ints
    crop_inds_h = (max_sample_h * torch.rand(*source_im.shape[:-3], num_crops).to(device)).long()
    crop_inds_w = (max_sample_w * torch.rand(*source_im.shape[:-3], num_crops).to(device)).long()
    crop_inds = torch.cat((crop_inds_h.unsqueeze(-1), crop_inds_w.unsqueeze(-1)), dim=-1)  # shape [..., N, 2]

    crops = crop_image_from_indices(
        images=source_im,
        crop_indices=crop_inds,
        crop_height=crop_height,
        crop_width=crop_width,
    )

    return crops, crop_inds


# N dimensional sample layer


# Linear layer which masks by "groups", with leftover features


# an empty layer, does nothing


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.


# coding: utf-8
# Code is adapted from:
# https://github.com/pclucas14/pixel-cnn-pp
# https://github.com/openai/pixel-cnn


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.zeros(tensor.size() + (n,), dtype=torch.float32, device=tensor.device)
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


def sample_from_discretized_mix_logistic(y, log_scale_min=-7.0,
                                         clamp_log_scale=False):
    """
    Sample from discretized mixture of logistic distributions

    Args:
        y (Tensor): B x C x T
        log_scale_min (float): Log scale minimum value

    Returns:
        Tensor: sample in range of [-1, 1].
    """
    assert y.size(1) % 3 == 0
    nr_mix = y.size(1) // 3

    # B x T x C
    y = y.transpose(1, 2)
    logit_probs = y[:, :, :nr_mix]

    # sample mixture indicator from softmax
    temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-5, 1.0 - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=-1)

    # (B, T) -> (B, T, nr_mix)
    one_hot = to_one_hot(argmax, nr_mix)
    # select logistic parameters
    means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
    log_scales = torch.sum(y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1)
    if clamp_log_scale:
        log_scales = torch.clamp(log_scales, min=log_scale_min)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = means.data.new(means.size()).uniform_(1e-5, 1.0 - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))

    x = torch.clamp(torch.clamp(x, min=-1.), max=1.)

    return x


# we can easily define discretized version of the gaussian loss, however,
# use continuous version as same as the https://clarinet-demo.github.io/


def sample_from_mix_gaussian(y, log_scale_min=-7.0):
    """
    Sample from (discretized) mixture of gaussian distributions
    Args:
        y (Tensor): B x C x T
        log_scale_min (float): Log scale minimum value
    Returns:
        Tensor: sample in range of [-1, 1].
    """
    C = y.size(1)
    if C == 2:
        nr_mix = 1
    else:
        assert y.size(1) % 3 == 0
        nr_mix = y.size(1) // 3

    # B x T x C
    y = y.transpose(1, 2)

    if C == 2:
        logit_probs = None
    else:
        logit_probs = y[:, :, :nr_mix]

    if nr_mix > 1:
        # sample mixture indicator from softmax
        temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-5, 1.0 - 1e-5)
        temp = logit_probs.data - torch.log(- torch.log(temp))
        _, argmax = temp.max(dim=-1)

        # (B, T) -> (B, T, nr_mix)
        one_hot = to_one_hot(argmax, nr_mix)

        # Select means and log scales
        means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
        log_scales = torch.sum(y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1)
    else:
        if C == 2:
            means, log_scales = y[:, :, 0], y[:, :, 1]
        elif C == 3:
            means, log_scales = y[:, :, 1], y[:, :, 2]
        else:
            assert False, "shouldn't happen"

    scales = torch.exp(log_scales)
    dist = D.Normal(loc=means, scale=scales)
    x = dist.sample()

    x = torch.clamp(x, min=-1.0, max=1.0)
    return x


def bias_zero_weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def zero_weight_init(m):
    """Custom weight init for Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


################################################################################


#################################################################################

class BranchedModules(nn.ModuleDict):
    def __init__(self, order=List[str], modules: Optional[Mapping[str, nn.Module]] = None, cat_dim=None,
                 split_sizes: List[int] = None, split_dim=None) -> None:
        super(BranchedModules, self).__init__(modules)
        assert modules is not None
        self._order = order
        self._cat_dim = cat_dim
        self._split_sizes = split_sizes
        self._split_dim = split_dim
        assert self._split_sizes is None or len(self._order) == len(self._split_sizes)
        assert self._split_sizes is None or all(s > 0 for s in self._split_sizes)
        assert self._split_sizes is None or split_dim is not None
        assert set(order) == set(modules.keys()), [order, modules.keys()]

    def forward(self, obs, ret_dict=False, **kwargs):
        all = AttrDict()
        all_ls = []

        if self._split_sizes is not None:
            obs_all = torch.split(obs, self._split_sizes, dim=self._split_dim)
        else:
            obs_all = [obs] * len(self._order)

        for k, o in zip(self._order, obs_all):
            all[k] = self[k](o, **kwargs)
            all_ls.append(all[k])

        if ret_dict:
            return all

        if self._cat_dim is not None:
            return torch.cat(all_ls, dim=self._cat_dim)

        return all_ls


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
               in root_module.named_modules(remove_duplicate=True)
               if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [k.split('.') for k, m
               in root_module.named_modules(remove_duplicate=True)
               if predicate(m)]
    assert len(bn_list) == 0
    return root_module


if __name__ == '__main__':
    linear_1 = torch.nn.Linear(in_features=1, out_features=10)
    linear_2 = torch.nn.Linear(in_features=10, out_features=10)
    linear_1.train()
    linear_2.train()

    in_arr = torch.ones(1, dtype=torch.float32)
    optim = torch.optim.Adam(list(linear_1.parameters()) + list(linear_2.parameters()), lr=1e-3)

    l1_out = linear_1.forward(in_arr)
    with torch_disable_grad(linear_2):
        l2_out = linear_2.forward(l1_out)

    optim.zero_grad()
    loss = (l2_out + 2).abs().mean()
    loss.backward()

    for param in linear_1.parameters():
        print("l1", param.grad)
    for param in linear_2.parameters():
        print("l2", param.grad)

    optim.zero_grad()
    l1_out_2 = linear_1.forward(in_arr)
    l2_out_2 = linear_2.forward(l1_out_2)
    (l2_out_2 - 2).abs().mean().backward()

    for param in linear_1.parameters():
        print("l1", param.grad)
    for param in linear_2.parameters():
        print("l2", param.grad)

    # combined usage
    optim.zero_grad()
    l1_out_3 = linear_1.forward(in_arr)
    with torch_disable_grad(linear_2):
        l2_out_3_ng = linear_2.forward(l1_out_3)
    l2_out_3 = linear_2.forward(l1_out_3)

    ls = (l2_out_3_ng + 2).abs().mean() + (l2_out_3 - 2).abs().mean()
    ls.backward()

    for param in linear_1.parameters():
        print("l1", param.grad)
    for param in linear_2.parameters():
        print("l2", param.grad)

    sys.exit(0)

    ## funcs
    arr = torch.arange(5 * 3 * 6).view((5, 3, 6))
    out = pad_dims(arr, [0, 2], [2, 4], delta=True, after=True, val=-1)
    out_np = pad_dims(to_numpy(arr), [0, 2], [2, 4], delta=True, after=True, val=-1)
    assert list(out.shape) == list(out_np.shape) == [7, 3, 10], out
    assert torch.equal(out[:-2, :, :-4], arr)
    assert np.alltrue(out_np[:-2, :, :-4] == to_numpy(arr))

    t1 = torch.arange(5)
    t2 = 10 * torch.ones(5)
    for i in range(1000):
        r = randint_between(t1, t2)
        assert torch.all(r >= t1) and torch.all(r < t2), r

    # like a big batch of data
    big_arr = np.empty((1024 * 256, 200, 30), dtype=np.float32)

    sizes = [(2,)] * 8 + [(4,)] + [(5,)] * 2
    keys = [f"key_{i}" for i in range(len(sizes))]
    names_to_shapes = AttrDict.from_kvs(keys, sizes)
    dtypes = [np.float32, np.uint8, np.float16] * 3 + [np.int, np.int]
    names_to_dtypes = AttrDict.from_kvs(keys, dtypes)

    with timeit("unconcatenate_copy"):
        out = unconcatenate(big_arr, keys, names_to_shapes, dtypes=names_to_dtypes, copy=True)
    print(timeit)
    timeit.reset()
    with timeit("unconcatenate_no_copy"):
        out2 = unconcatenate(big_arr, keys, names_to_shapes, dtypes=names_to_dtypes, copy=False)
    print(timeit)
    timeit.reset()
    with timeit("unconcatenate_no_copy_out"):
        out3 = unconcatenate(big_arr, keys, names_to_shapes, dtypes=names_to_dtypes, copy=False, outs=out2)
    print(timeit)
    timeit.reset()
