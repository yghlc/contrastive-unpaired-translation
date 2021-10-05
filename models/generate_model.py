"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from .base_model import BaseModel
from . import networks


class GenerateModel(BaseModel):
    '''
    This GenerateModel can be used to generate GAN results for only one direction from A to B, for satellite images
    This model will automatically set '--dataset_mode generate', which only loads the images from one collection.
    See the test instruction for more details.
    '''
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        assert not is_train, 'GenerateModel cannot be used during training time'
        parser.set_defaults(dataset_mode='generate')
        # parser.add_argument('--model_suffix', type=str, default='',
        #                     help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        assert (not opt.isTrain)
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []

        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['fake']

        # specify the models you want to use
        # you can use opt to specify different behaviors for training and test.
        # CUT
        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids,
                                      opt)
        # cycle GAN
        # self.model_names = ['G_A', 'G_B'] # Cycle GAN

        # # assigns the model to self.netG_[suffix] so that it can be loaded
        # # please see <BaseModel.load_networks>
        # setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        # self.set_input(data)
        # bs_per_gpu = self.real.size(0) // max(len(self.opt.gpu_ids), 1)
        # self.real = self.real[:bs_per_gpu]
        # self.forward()  # compute fake images: G(A)
        pass

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.real = input['A'].to(self.device)
        self.img_idx = input['img_idx']
        self.tile_idx = input['tile_idx']
        self.tile_obj = input['tile_obj']

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.fake = self.netG(self.real)  # G(real)

    def optimize_parameters(self):
        """No optimization for test model."""
        pass

    def get_img_tile_info(self):
        return self.img_idx, self.tile_idx,self.tile_obj
