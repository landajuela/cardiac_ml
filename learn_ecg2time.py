import sys
import configparser
import datetime
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.optim import lr_scheduler
from torchsummary import summary
from cardiac_ml.data_interface import read_data_dirs
from cardiac_ml.data_interface import Ecg2TimeDataset
from cardiac_ml.trainable_model import TrainableModel
from cardiac_ml.io_util import CommandParser, OutputHandler
from cardiac_ml.ml_util import init_weights, get_name_parameters

# Squeeze net
class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv1d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv1d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv1d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
            ], 1)

class SqueezeNet(nn.Module):

    def __init__(self, version='1_0', dropout = 0.5, kernel_size = 3):
        super(SqueezeNet, self).__init__()
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv1d(12, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool1d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool1d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == '1_1':  # PAPER : In the original PAPER kernel_size = 3
            self.features = nn.Sequential(
                nn.Conv1d(12, 64, kernel_size=kernel_size, stride=2, padding = 1),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=kernel_size, stride=2, padding = 1, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool1d(kernel_size=kernel_size, stride=2, padding = 1, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool1d(kernel_size=kernel_size, stride=2, padding = 1, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256)
            )
        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv1d(512, 75, kernel_size=3, stride=2, padding = 0)
       
        final_conv_II = nn.Conv1d(75, 75, kernel_size=3, stride=2, padding = 0)
        final_conv_III = nn.Conv1d(75, 75, kernel_size=3, stride=2, padding = 0)
        self.classifier = nn.Sequential(
            nn.Dropout(p = dropout),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(3),
            final_conv_III
        )
    
        #Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__=='__main__':
    
    # Read the parse arguments (file name)
    commandParser = CommandParser()
    commandArgs = commandParser.get_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('The device is {}\n'.format(device))

    # Parsing input data file
    config = configparser.ConfigParser()
    config.read(commandArgs.file)
    
    # Parse a list of directories with data
    datapaths_train = config['DATA']['datapaths_train']
    dirs_names_train = [str(_).strip() for _ in datapaths_train.split(',')]
    print('Training files:  ',end='')
    file_pairs_train = read_data_dirs(dirs_names_train)
    datapaths_val = config['DATA']['datapaths_val']
    dirs_names_val = [str(_).strip() for _ in datapaths_val.split(',')]
    print('Validation files:  ',end='')
    file_pairs_val = read_data_dirs(dirs_names_val)
    print('')
    data_scaling_ecg =  config['DATA']['data_scaling_ecg']
    initial_time_aug =  config['DATA']['initial_time_aug']
    data_scaling_vm =  config['DATA']['data_scaling_vm']
            
    num_timesteps = int(config['DATA']['num_timesteps'])
    all_channels = config['DATA'].getboolean('all_channels')
    if all_channels:
        input_dim = int(config['DATA']['input_dim'])     # input dimension
        output_dim = int(config['DATA']['output_dim'])   # output dimension  
        channels_in = list(range(input_dim))
        channels_out = list(range(output_dim))
    else:        
        tmp_in = config['DATA']['channels_in']
        tmp_out = config['DATA']['channels_out']
        channels_in = [int(_) for _ in tmp_in.split(',')]
        channels_out = [int(_) for _ in tmp_out.split(',')]
        input_dim = len(channels_in)
        output_dim = len(channels_out)
    
    # Model
    hidden_dim = int(config['MODEL']['hidden_dim'])   # hidden layer dimension
    layer_dim = int(config['MODEL']['layer_dim'])    # number of hidden layers

    # Train
    learning_rate = float(config['TRAIN']['learning_rate'])
    gamma_scheduler = float(config['TRAIN']['gamma_scheduler'])
    step_size_scheduler = float(config['TRAIN']['step_size_scheduler'])
    batch_size = int(config['TRAIN']['batch_size'])
    num_epochs = int(config['TRAIN']['num_epochs'])
    grad_clippling = config['TRAIN'].getboolean('grad_clippling')
    dropout = float(config['TRAIN']['dropout'])
    loss_norm = config['TRAIN']['loss_norm']
    load_model = config['TRAIN'].getboolean('load_model')
    model_path = config['TRAIN']['model_path']
    
    # Save
    outputstats_file = config['SAVE']['outputstats_file']
    checkpoint_rate = int(config['SAVE']['checkpoint_rate'])
    out_name = config['SAVE']['out_name']
    progressbar = config['SAVE'].getboolean('progressbar')
        
    # Overwrite with terminal inputs
    if not commandArgs.type == None:
        rnn_type = commandArgs.type
        print('Detected command-line argument : rnn_type = {}\n'.format(rnn_type))    
    
    if not commandArgs.hdim == None:
        hidden_dim = commandArgs.hdim
        print('Detected command-line argument : hidden_dim = {}\n'.format(hidden_dim))

    if not commandArgs.ldim == None:
        layer_dim = commandArgs.ldim
        print('Detected command-line argument : layer_dim= {}\n'.format(layer_dim))

    if not commandArgs.sqlen == None:
        seq_len = commandArgs.sqlen
        print('Detected command-line argument : seq_len = {}\n'.format(seq_len))    

    if not commandArgs.lr == None:
        learning_rate = commandArgs.lr
        print('Detected command-line argument : learning_rate = {}\n'.format(learning_rate))

    if not commandArgs.gm == None:
        gamma_scheduler = commandArgs.gm
        print('Detected command-line argument : gamma_scheduler = {}\n '.format(gamma_scheduler))

    if not commandArgs.clip == None:
        grad_clippling = commandArgs.clip
        print('Detected command-line argument : grad_clippling = {}\n'.format(grad_clippling))

    if not commandArgs.norm == None:
        loss_norm = commandArgs.norm
        print('Detected command-line argument : loss_norm = {}\n'.format(loss_norm))
    
    if not commandArgs.bar == None:
        progressbar = commandArgs.bar
        print('Detected command-line argument : progressbar = {}\n'.format(progressbar))    
        
    nsecg = 'none'    
    if not commandArgs.nsecg == None:
        nsecg = commandArgs.nsecg
        print('Detected command-line argument : nsecg = {}\n'.format(nsecg))
    
    nsvm = 'none'  
    if not commandArgs.nsvm == None:
        nsvm = commandArgs.nsvm
        print('Detected command-line argument : nsvm = {}\n'.format(nsvm))
        
    knl = 3
    if not commandArgs.knl == None:
        knl = commandArgs.knl
        print('Detected command-line argument : kernel = {}\n'.format(knl)) 
        
    if not commandArgs.dpo == None:
        dropout = commandArgs.dpo
        print('Detected command-line argument : dpo = {}\n'.format(dropout)) 

    # PRINT THE ACTUAL DATA
    now = datetime.datetime.now()
    now = str(now.year)+str(now.month).zfill(2)+str(now.day).zfill(2)+'-'+str(now.hour).zfill(2)+str(now.minute).zfill(2)+str(now.second).zfill(2)
    sample = open('hyperparams_command_parsed_'+now+'.config', 'w')
    for section in config.sections():
        print('[' + section + ']', file = sample)
        for name, value in config.items(section):
            print(' {} = {}'.format(name, vars()[name]), file = sample)
        print('', file = sample)
    sample.close()
    
    # Dataset
    print('Training data:  ',end='') 
    trainingData = Ecg2TimeDataset(file_pairs_train, channels_out, num_timesteps, data_scaling_ecg, data_scaling_vm, initial_time_aug, nsecg, nsvm)    
    trainingData = data.DataLoader(trainingData, shuffle = True, batch_size = batch_size, num_workers=20)
    print('Lenght of training dataloader (batches) : {} \n'.format(len(trainingData)))
    print('Validation data:  ',end='') 
    valData = Ecg2TimeDataset(file_pairs_val, channels_out, num_timesteps, data_scaling_ecg, data_scaling_vm, False, 'none', 'none')
    valData = data.DataLoader(valData, shuffle = True, batch_size = batch_size, num_workers=20)
    print('Lenght of validation dataloader (batches) : {} \n'.format(len(valData)))

    # Models   
    if load_model:
        # network_dir = './split_data_hearts_dd_0p2_20190913_95_5_20190920'
        # model_name = network_dir + '/'+ 'mymodel.best.pth'
        print('Loading model : {}'.format(model_path))
        model = torch.load(model_path)
    else:
        model = SqueezeNet(version='1_1', dropout=dropout, kernel_size = knl)    
        
    model = model.to(device)
    # Print model summary to file
    orig_stdout = sys.stdout
    f = open('model_summary.txt', 'w')
    sys.stdout = f
    print('print(model):\n')
    print(model)
    print('\n')
    print('summary(model,shape):\n')
    summary(model,(12,num_timesteps))
    sys.stdout = orig_stdout
    f.close() 
    
    if not load_model:       
        model.apply(init_weights)
    
    # Output handler
    metadata_err = 'epoch, trainingLoss, validationLoss'
    name_parameters = get_name_parameters(model)
    
    num_parameters = len(list(model.parameters()))
    name_parameters = name_parameters[0:num_parameters]

    metadata_grad = ', '.join(name_parameters)
    outputHandler = OutputHandler(outputstats_file, metadata_err, (5,), metadata_grad, (len(name_parameters),)) 
    
    ## Loss
    if loss_norm == 'MSE':
        criterion = nn.MSELoss()
    elif loss_norm == 'L1':
        criterion = nn.L1Loss()

    ## Optimizer
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    ## Scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size = step_size_scheduler, gamma = gamma_scheduler)

    ## Train loop
    tt = TrainableModel(criterion, optimizer, scheduler, outputHandler, device, progressbar)
    tt.learn(model = model, trainingData = trainingData, valData = valData, 
             num_epochs = num_epochs, grad_clippling = grad_clippling, checkpointRate = checkpoint_rate, name = out_name)

    
    
    
