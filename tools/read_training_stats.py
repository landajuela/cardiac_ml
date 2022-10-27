#!/usr/bin/env python3
import sys
import h5py
import argparse
import matplotlib.pyplot as plt

# Function for Boolean type in the arguments
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def myargparse():
    description = 'Plot error evolution for learnt models.\
    Usage : python3 ~/PyTorch/tools/read_training_stats.py -out bestloss -print_style hyper_tuning -plot False'
    parser = argparse.ArgumentParser(description=description,
                                     epilog='End of documentation')
    parser.add_argument('file',
                       help='h5 file to read data from')
    parser.add_argument('-out',
                        dest='out',            
                        help="Name of output file.",
                        default='bestLoss')
    parser.add_argument('-print_style', 
                        type=str,
                        dest='print_style',            
                        help="Print style. Options: hyper_tuning, train_val, all (Default: hyper_tuning)",
                        default='hyper_tuning')
    parser.add_argument('-plot','--plot_results',  
                        type=str2bool,
                        dest='plot',            
                        help="Plot the error evolution (Default: False).",
                        default=False)  
    parser.add_argument('-print_screen',  
                        type=str2bool,
                        dest='print_screen',            
                        help="Show plots in screen",
                        default=False)                                                                                              
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    return parser.parse_args()
    
if __name__ == '__main__':
    
    args = myargparse()
    
    f_errors = h5py.File(args.file,'r')
    dataImportErr = f_errors['dataset']
    # Recall that:
    # self.outputHandler.write_errors(epoch + 1, num_epochs, trainingLoss, validationLoss, bestLoss, bestEpoch + 1 )
    # def write_errors(self, epoch, num_epochs, trainingLoss, validationLoss, bestLoss, bestEpoch):
    #     self._hdf5_errors.append(np.array([epoch, trainingLoss, validationLoss, bestLoss, bestEpoch]))
    # Thus,
    # dataImportErr[i] = [epoch, trainingLoss, validationLoss, bestLoss, bestEpoch]
    
    print('Columns in h5 file : {}'.format(dataImportErr.shape[1]))
    print('Current number of errors computed : {}'.format(dataImportErr.shape[0]))
    
    out_file = open(args.out + '.out','w')
        
    # bestLoss is a validation loss    
    if args.print_style == 'hyper_tuning' :
        bestLoss = dataImportErr[-1,3]
        bestEpoch = int(dataImportErr[-1,4])
        print(f'{bestEpoch}  {bestLoss}', file = out_file)
    if args.print_style == 'train_val' :
        bestLoss = dataImportErr[-1,3]
        bestEpoch = int(dataImportErr[-1,4])
        trainLossInbestEpoch = dataImportErr[int(bestEpoch-1),1]
        print('Best epoch for validation:{:3d}, Validation loss:{:11.4f}, Training loss:{:11.4f}'.format(bestEpoch, bestLoss, trainLossInbestEpoch), file = out_file)                        
    if args.print_style == 'all':
        for i in range(dataImportErr.shape[0]):
            print('Epoch {:3d},  training : {:11.4f}   validation : {:11.4f}   best validation loss : {:11.4f} on epoch {}'.format(int(dataImportErr[i,0]), dataImportErr[i,1], dataImportErr[i,2], dataImportErr[i,3], int(dataImportErr[i,4])), file = out_file)
            
    out_file.close
    
    if args.plot:
        # Plot the errors
        plt.figure(figsize=(8, 8))
        plt.loglog(dataImportErr[:,0], dataImportErr[:,2], label='valError')
        plt.loglog(dataImportErr[:,0], dataImportErr[:,1], label='trainError')
        plt.loglog(dataImportErr[:,0], dataImportErr[:,3], label='bestLoss')
        # plt.loglog(dataImportErr[:,0], dataImportErr[:,4], label='bestEpoch')
        # plt.title('Error evolution (' + 'learning_rate: ' + str(learning_rate) + ', batch_size: ' + str(batch_size) + ', \n grad_clippling: ' +  str(grad_clippling) + ', dropout: ' + str(dropout) + ', \n loss_norm: ' + str(loss_norm) + ', data_scaling: ' + str(data_scaling) +')')
        plt.ylabel('Error')
        plt.xlabel('Epoch')
        plt.legend()
        if args.print_screen:
            plt.show()
        plt.savefig('plot_error.pdf', dpi=200)
    
        # # Plot the gradients
        # plt.figure(figsize=(8, 8))
        # for i in range(dataImportGrad.shape[1]):
        #     plt.loglog(dataImportGrad[:,i], label = name_parameters[i])
        # plt.title('Grad evolution')
        # plt.ylabel('Norm')
        # plt.xlabel('Epoch')
        # plt.legend()
        # if args.print_screen:
        #     plt.show()
        # plt.savefig(args.base_name + '_plot_gradients.pdf', dpi=150)       

    
    f_errors.close()        
    
    
    
    
    
