import os
import math


class NetOption(object):

    def __init__(self):
        #  ------------ General options ---------------------------------------
        self.save_path = ""  # log path
        self.data_path = "/usr/home2/code/darklight_concat/as-oct/"  # path for loading data set  \
        self.label_path = "/usr/home2/code/darklight_concat/as-oct/DL_label11_11.csv"
        self.data_set = "brightVsDark"  # options: asoct | brightVsDark
        self.disease_type = 1 # 1(open) | 2(narrow) | 3(close) | 4(unclassify)  or  1(open) | 2(narrow/close)
        self.manualSeed = 1  # manually set RNG seed
        self.nGPU = 1  # number of GPUs to use by default
        self.GPU = 3  # default gpu to use, options: range(nGPU)
        self.datasetRatio = 1.0  # greedy increasing training data for cifar10
        self.no_cuda = False
        # ------------- Data options ------------------------------------------
        self.nThreads = 10  # number of data loader threads

        # ------------- Training options --------------------------------------
        self.testOnly = False  # run on validation set only
        self.tenCrop = False  # Ten-crop testing

        # ---------- Optimization options -------------------------------------
        self.nEpochs = 80  # number of total epochs to train
        self.batchSize = 4  # mini-batch size
        self.LR = 0.001  # initial learning rate
        self.lrPolicy = "multistep"  # options: multistep | linear | exp
        self.momentum = 0.9  # momentum
        self.weightDecay = 1e-4  # weight decay 1e-2
        self.gamma = 0.94  # gamma for learning rate policy (step)
        self.step = 2.0  # step for learning rate policy
        self.nesterov = True
        # ---------- Model options --------------------------------------------
        self.trainingType = 'onevsall'# options: onevsall | multiclass
        self.model = "resnet3D"
        # self.netType = "ResNet3D"  # options: ResNet | DenseNet | Inception-v3 | AlexNet | ResNet3D
        self.experimentID = "new_label_darkvslight_1111_grayimage_noPretrain_onlyNarrow_minusInLayer4_noDataAug"
        self.model_depth = 50  # resnet depth: (n-2)%6==0
        self.wideFactor = 1  # wide factor for wide-resnet
        self.resnet_shortcut = 'B'  ##'Shortcut type of resnet (A | B)')
        self.sample_size = 112  ##Height and width of inputs
        self.sample_duration = 16  #Temporal duration of inputs   输入的时间期限  关注一下
        # ---------- Resume or Retrain options --------------------------------
        ##v3

        # self.model1 = '/home/yangyifan/jingwen_code_oct_v2/as-oct/log_asoct_ResNet_50_onevsall_bs16_type=2-lr=0.001/model/best_model.pkl'## load model for output
        # self.retrain = [self.model1]  # path to model to retrain with
        self.retrain = None # path to model to retrain with
        self.resume = None
        # self.resume = "/usr/home2/code/dark_light_128_master/as-oct/log_asoct_ResNet_18_onevsall_bs8_darkvslight_1107_grayimage_noPretrain_onlyNarrow_minusInLayer1_noDataAug/model/checkpoint60.pkl"  # path to directory containing checkpoint
        # self.resume = "/usr/home2/code/jingwen_code_oct_cropped/as-oct/log_asoct_ResNet_18_onevsall_bs8_addpad_9.6_3foldType=1-lr=0.01/model/checkpoint7.pkl"
        self.resumeEpoch =0   # manual epoch number for resume
        # self.pretrain = 'pretrain/resnet18.pth'
        self.pretrain = False
        self.n_finetune_classes = 400 ## 'Number of classes for fine-tuning. n_classes is set to the number when pretraining.'

        # check parameters
        self.paramscheck()

    def paramscheck(self):
        self.save_path = "log_%s_%s_%d_%s_bs%d_%s/" % \
        (self.data_set, self.model,
            self.model_depth, self.trainingType, self.batchSize, self.experimentID)
        if self.data_set == 'brightVsDark':
            if self.trainingType == 'onevsall':
                self.n_classes = 1
            else:
                self.n_classes = 4
            self.ratio = [1.0/2, 2.7/3]
