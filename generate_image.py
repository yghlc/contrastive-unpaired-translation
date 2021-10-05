"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md

# copy and modify from 'test.py' for generate satellite images (one side only)
# By Lingcao Huang
# Octorber 4, 2021

"""
import os,sys
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util

deeplabforRS =  os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, deeplabforRS)

import basic_src.io_function as io_function
import raster_io

def save_satellite_tile(img_idx, tile_idx, boundary, org_img,res_dir, visuals):
    '''
    save a genterate tile of satellite images
    Parameters:
        img_idx                  -- index of satellite image
        tile_idx                 -- index of the tile within a satellite image
        boundary                 -- the extent of the path: (xoff,yoff ,xsize, ysize)
        org_img                  -- the original image path
        res_dir                  -- the result directory
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs

    This function will save images stored in 'visuals' to dis,
    '''

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        # image_name = '%s/%s.png' % (label, name)
        image_dir = os.path.join(res_dir, 'I%d' % img_idx)
        if os.path.isdir(img_idx) is False:
            io_function.mkdir(image_dir)
        save_path = os.path.join(image_dir, 'I%d_%d.tif' % (img_idx, tile_idx))
        raster_io.save_numpy_array_to_rasterfile(im, save_path, org_img, format='GTiff',boundary=boundary)



if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(util.copyconf(opt, phase="generate"))  # create a dataset given opt.dataset_mode and other options

    model = create_model(opt)      # create a model given opt.model and other options

    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        # if i >= opt.num_test:  # only apply our model to opt.num_test images.
        #     break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        # img_path = model.get_image_paths()     # get image paths
        img_idx, tile_idx, tile_boundary, org_img = model.get_img_tile_info()
        # if i % 5 == 0:  # save images to an HTML file
        print('processing (%04d)-th image (%05d)-th tiles...' % (img_idx, tile_idx))
        # save_images(webpage, visuals, img_path, width=opt.display_winsize)
        save_satellite_tile(img_idx,tile_idx,tile_boundary,org_img,opt.results_dir,visuals)
    # webpage.save()  # save the HTML
