"""Dataset class SatelliteImage

You can specify '--dataset_mode SatelliteImage' to use this dataset.

You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""

import os,sys

deeplabforRS =  os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, deeplabforRS)

import basic_src.io_function as io_function
import split_image
import raster_io

from datetime import datetime

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset


# for testing
from PIL import Image
import random
import util.util as util

import numpy as np

# copy from build_RS_data.py
class patchclass(object):
    """
    store the information of each patch (a small subset of the remote sensing images)
    """
    def __init__(self,org_img,boundary):
        self.org_img = org_img  # the original remote sensing images of this patch
        self.boundary=boundary      # the boundary of patch (xoff,yoff ,xsize, ysize) in pixel coordinate
    def boundary(self):
        return self.boundary


def get_satellite_img_list(input_str, extension):
    '''get file list from a dirctory or txt file'''
    if os.path.isdir(input_str):
        img_list = io_function.get_file_list_by_pattern_ls(extension,input_str)
    elif os.path.isfile(input_str):
        img_list = io_function.read_list_from_txt(input_str)
    else:
        raise ValueError('unknow input option for satellite image list: %s'%str(input_str))
    return img_list


def split_satellite_img_to_tiles(simg_list, save_dir, tile_width, tile_height, adj_overlay_x=0,adj_overlay_y=0,out_format='PNG'):
    if os.path.isdir(save_dir) is False:
        io_function.mkdir(save_dir)

    tile_list = []
    for idx,img_path in enumerate(simg_list):
        print(datetime.now(),'split %d / %d, %s to tiles '%(idx, len(simg_list), img_path))
        out_list = split_image.split_image(img_path, save_dir, tile_width, tile_height, adj_overlay_x, adj_overlay_y, out_format)
        tile_list.extend(sorted(out_list))
    
    # check black tiles or most part of the sub-images is black (nodata)
    new_tile_list = []
    delete_tile_list = []
    tile_dir_delete = save_dir + '_delete'
    io_function.mkdir(tile_dir_delete)

    for tile_path in tile_list:
        valid_per, entropy = raster_io.get_valid_percent_shannon_entropy(tile_path)    # base=10
        if valid_per > 60 and entropy >= 0.5:
            new_tile_list.append(tile_path)
        else:
            delete_tile_list.append(tile_path)
            io_function.movefiletodir(tile_path,tile_dir_delete)

    tile_list = new_tile_list
    return tile_list


def read_one_image_tile(tile_list, index):
    img_idx, patch_obj, patch_idx = tile_list[index]['img_idx'], tile_list[index]['img_tile'], tile_list[index]['tile_idx']

    tile_data, nodata = raster_io.read_raster_all_bands_np(patch_obj.org_img,boundary = patch_obj.boundary)
    if nodata is not None:
        tile_data[np.where(tile_data == nodata)] = 0    # replace nodata as 0

    band_count = tile_data.shape[0]
    if band_count == 3:
        color_mode = 'RGB' # # need consider one band or others
    else:
        color_mode = ''
        raise ValueError('Currently, only support 3-band')

    # because TORCHVISION.TRANSFORMS accept both PIL images and tensor images, so we convert numpy image to PIL image.
    data_rasterio_pil = tile_data.transpose(1,2,0)  # from (C,H,W) to (H, W, C)
    tile_data_pil = Image.fromarray(data_rasterio_pil,mode=color_mode)
    return tile_data_pil, img_idx, patch_idx, patch_obj


def test_read_one_image_tile():
    '''testing read image data through rasterio'''
    # also compare with the data reading through PIL


    image_path = os.path.expanduser('~/Data/tmp_data/CUT_dataset_test/trainA/20200818_mosaic_8bit_rgb_p_22.png')

    data_rasterio,nodata = raster_io.read_raster_all_bands_np(image_path)

    # read_one_image_tile(tile_list, index)

    # reading the image using PIL
    A_img = Image.open(image_path).convert('RGB')
    print(A_img)
    # A = self.transform(A_img)
    print(data_rasterio.shape)  # channel, height, width
    print(A_img.size, A_img.mode)

    # PIL image to numpy image
    pil_np_data = np.asarray(A_img).transpose(2,0,1)       # to numpy, from (H, W, C) to (C,H,W)
    print(pil_np_data.shape)

    diff = data_rasterio - pil_np_data
    print('diff sum:',np.sum(diff))
    print('diff max:',np.max(diff))
    print('diff min:',np.min(diff))

    # convert numpy image to PIL image and save to disk
    save_path = io_function.get_name_by_adding_tail(image_path,'numpy_to_PIL')
    print(data_rasterio.dtype)
    data_rasterio_pil = data_rasterio.transpose(1,2,0)  # from (C,H,W) to (H, W, C)
    np_pil = Image.fromarray(data_rasterio_pil,mode='RGB')
    print(np_pil.size, np_pil.mode)
    np_pil.save(save_path)



class SatelliteImageDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--tile_width', type=int, default=1024, help='the width of image tile, in pixels')
        parser.add_argument('--tile_height', type=int, default=1024, help='the height of image tile, in pixels')
        parser.add_argument('--overlay_x', type=int, default=0, help='the overlay of adjacent tiles for x direction, in pixels')
        parser.add_argument('--overlay_y', type=int, default=0, help='the overlay of adjacent tiles for y direction, in pixels')

        parser.add_argument('--image_A_dir_txt', type=str, help='the directory or list (txt) of images in domain A (src), '
                                                                'if is dir, it will get all images in the folder (not sub-fodler)')
        parser.add_argument('--image_B_dir_txt', type=str, help='the directory or list (txt) of images in domain B (target), '
                                                                'if is dir, it will get all images in the folder (not sub-fodler),'
                                                                'if test, we can ignore this')
        parser.add_argument('--extension', type=str,default='.tif', help='the extension of image files')

        parser.add_argument('--work_dir', type=str,default='./', help='the working directory, will save some intermediate files')

        # parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root

        BaseDataset.__init__(self, opt)
        # dirctory for saving image tiles and their path for training
        self.dir_A = None
        self.dir_B = None
        self.A_paths = None
        self.B_paths = None
        self.is_generate = opt.phase == "generate"

        self.tiles_for_images = []  # this is 2d list, each 1d list is tiles for one images
        self.tiles_1d = []          # this is 1d list of {img_index, tile}

        if opt.phase == "generate":
            # generating GAN results only for one side with the model option '-model test'
            sate_img_list = get_satellite_img_list(opt.image_A_dir_txt, opt.extension)
            if len(sate_img_list) > opt.max_dataset_size:
                print(datetime.now(), 'warning, the count of images (%d) to translate is greater than max_dataset_size, '
                                      'will only translate the first %d ones'%(len(sate_img_list), opt.max_dataset_size))
                sate_img_list = sate_img_list[:opt.max_dataset_size]

            for img_idx, img_path in enumerate(sate_img_list):
                height, width, b_count, dtypes = raster_io.get_height_width_bandnum_dtype(img_path)
                # split the image
                patch_boundary = split_image.sliding_window(width, height, opt.tile_width,opt.tile_height,
                                                            opt.overlay_x,opt.overlay_y)
                patches_of_a_image = []
                for t_idx, patch in enumerate(patch_boundary):
                    # need to handle the patch with smaller size
                    # if patch[2] < crop_width or patch[3] < crop_height:   # xSize < 480 or ySize < 480
                    #     continue
                    img_patch = patchclass(img_path, patch)
                    patches_of_a_image.append(img_patch)

                    self.tiles_1d.append({'img_idx':img_idx, 'img_tile':img_patch, 'tile_idx':t_idx})

                print('img_idx:',img_idx,img_path,'is divided to',len(patches_of_a_image), 'tiles')
                self.tiles_for_images.append(patches_of_a_image)

            input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
            self.transform = get_transform(opt, grayscale=(input_nc == 1))

        else:
            self.dir_A = os.path.join(opt.work_dir, opt.phase + 'A')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opt.work_dir, opt.phase + 'B')  # create a path '/path/to/data/trainB'

            if opt.phase == "test" and not os.path.exists(self.dir_A) \
                    and os.path.exists(os.path.join(opt.work_dir, "valA")):
                self.dir_A = os.path.join(opt.work_dir, "valA")
                self.dir_B = os.path.join(opt.work_dir, "valB")

            # get satellite image list
            sate_img_list_A = get_satellite_img_list(opt.image_A_dir_txt,opt.extension)
            sate_img_list_B = get_satellite_img_list(opt.image_B_dir_txt,opt.extension)

            # convert the satellite images to many patches (tiles) if it for training and not existing.
            if os.path.exists(self.dir_A) and os.path.exists(self.dir_B):
                self.A_paths = sorted(
                    make_dataset(self.dir_A, opt.max_dataset_size))  # load images from 'trainA'
                self.B_paths = sorted(
                    make_dataset(self.dir_B, opt.max_dataset_size))  # load images from 'trainB'
            else:
                # split images to tiles
                self.A_paths = split_satellite_img_to_tiles(sate_img_list_A,self.dir_A,opt.tile_width,opt.tile_height,
                                                            opt.overlay_x,opt.overlay_y)
                self.B_paths = split_satellite_img_to_tiles(sate_img_list_B,self.dir_B,opt.tile_width,opt.tile_height,
                                                            opt.overlay_x,opt.overlay_y)

                self.A_paths = self.A_paths[:min(opt.max_dataset_size, len(self.A_paths))]
                self.B_paths = self.B_paths[:min(opt.max_dataset_size, len(self.B_paths))]

            self.A_size = len(self.A_paths)  # get the size of dataset A
            self.B_size = len(self.B_paths)  # get the size of dataset B
            io_function.save_list_to_txt('ready_to_train.txt',['image count: %d \n'%item for item in [self.A_size,self.B_size]])



    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

       Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        if self.is_generate:
            # A_path = self.A_paths[index]
            # A_img = Image.open(A_path).convert('RGB')
            # A = self.transform(A_img)
            # return {'A': A, 'A_paths': A_path}
            A_img, img_idx, tile_idx, tile_obj = read_one_image_tile(self.tiles_1d, index)
            A = self.transform(A_img)
            # got error: TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists;
            # found <class 'data.satelliteimage_dataset.patchclass'>
            # change to not using patchclass
            return {'A': A, 'img_idx': img_idx, 'tile_idx': tile_idx,'tile_boundary':tile_obj.boundary, 'org_img':tile_obj.org_img}

        else:
            # copy from unaligned_dataset.py
            A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
            if self.opt.serial_batches:   # make sure index is within then range
                index_B = index % self.B_size
            else:   # randomize the index for domain B to avoid fixed pairs.
                index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')

            # Apply image transformation
            # For FastCUT mode, if in finetuning phase (learning rate is decaying),
            # do not perform resize-crop data augmentation of CycleGAN.
    #        print('current_epoch', self.current_epoch)
            is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
            modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
            transform = get_transform(modified_opt)
            A = transform(A_img)
            B = transform(B_img)

            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of all tiles of images."""
        if self.is_generate:
            total_count = 0
            for tiles in self.tiles_for_images:
                total_count += len(tiles)
            return total_count
        else:
            return max(len(self.A_paths), len(self.B_paths))

