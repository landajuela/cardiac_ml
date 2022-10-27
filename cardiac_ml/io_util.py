import argparse, sys
import numpy as np
import h5py

# Parse the arguments from command line

# Function for Boolean type in the arguments in argparse
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class CommandParser:
    def __init__(self):     
        description = 'Learn a model.'
        epilog = 'End of documentation'
        self._parser = argparse.ArgumentParser(description=description,
                                               epilog=epilog)
        self._parser.add_argument('file',      
                                help='file to read data from',
                                default='hyperparams.config')
        self._parser.add_argument('-type','--rnn_type',
                        dest='type',            
                        help='Type of Architecture: RNN, GRU, LSTM, Linear, Nonlinear.')                          
        self._parser.add_argument('-hdim','--hidden_dim',
                        dest='hdim',            
                        help='Hidden dimension.',
                        type=int)
        self._parser.add_argument('-ldim','--layer_dim',
                        dest='ldim',            
                        help='Layer dimension.',
                        type=int)
        self._parser.add_argument('-sqlen','--seq_len',
                        dest='sqlen',            
                        help='Sequence length.',
                        type=int)                                       
        self._parser.add_argument('-lr','--learning_rate',
                        dest='lr',            
                        help='Value of learning rate.',
                        type=float)
        self._parser.add_argument('-wd','--weight_decay',
                        dest='wd',            
                        help='Value of weight decay.',
                        type=float)
        self._parser.add_argument('-gm','--gamma_scheduler',
                        dest='gm',            
                        help='Value of gamma scheduler.',
                        type=float)
        self._parser.add_argument('-clip', '--grad_clippling',
                        type=str2bool,
                        dest='clip',            
                        help='Clippling or not clippling.')
        self._parser.add_argument('-norm', '--loss_norm',
                        type=str,
                        dest='norm',            
                        help='Which norm to consider in the loss.')
        self._parser.add_argument('-bar', '--progressbar',
                        type=str2bool,
                        dest='bar',            
                        help='Print progressbar in terminal.')
        self._parser.add_argument('-nsecg','--noise_ecg',
                        dest='nsecg',            
                        help='Value of noise_ecg.',
                        type=float)
        self._parser.add_argument('-nsvm','--noise_vm',
                        dest='nsvm',            
                        help='Value of noise_vm.',
                        type=float)
        self._parser.add_argument('-knl','--kernel_size',
                    dest='knl',            
                    help='Value of kernel_size.',
                    type=int)
        self._parser.add_argument('-dpo','--dropout',
                    dest='dpo',            
                    help='Value of dropout.',
                    type=float)                                                                                                                    
        if len(sys.argv) == 1:
            self._parser.print_help()
            sys.exit(1)
            
    def get_args(self):
        return self._parser.parse_args()

class HDF5Store(object):
    """
    Simple class to append value to a hdf5 file on disc (usefull for building keras datasets)
    
    Params:
        datapath: filepath of h5 file
        dataset: dataset name within the file
        shape: dataset shape (not counting main/batch axis)
        dtype: numpy dtype
    
    Usage:
        hdf5_store = HDF5Store('/tmp/hdf5_store.h5','X', shape=(20,20,3))
        x = np.random.random(hdf5_store.shape)
        hdf5_store.append(x)
        hdf5_store.append(x)
        
    From https://gist.github.com/wassname/a0a75f133831eed1113d052c67cf8633
    """
    def __init__(self, datapath, dataset, shape, dtype=np.float32, compression="gzip", chunk_len=1):
        self.datapath = datapath
        self.dataset = dataset
        self.shape = shape
        self.i = 0
        
        h5f =  h5py.File(self.datapath, mode='w')       
        dset = h5f.create_dataset(
            self.dataset,
            shape=(0, ) + shape,
            maxshape=(None, ) + shape,
            dtype=dtype,
            compression=compression,
            chunks=(chunk_len, ) + shape)
        # ADDED
        h5f.close()
    
    def append(self, values):
        
        h5f =  h5py.File(self.datapath, mode='a')
        dset = h5f[self.dataset]
        dset.resize((self.i + 1, ) + self.shape)
        dset[self.i] = [values]
        self.i += 1
        #h5f.flush()
        # ADDED
        h5f.close()
            
    def write_attrs(self, obj):
        
        h5f =  h5py.File(self.datapath, mode='a')
        dset = h5f[self.dataset]
        dset.attrs['metadata'] = obj
        # ADDED
        #h5f.flush()
        h5f.close()                
        
class OutputHandler:
    
    def __init__(self, name, metadata_errors, shape_errors, metadata_grads, shape_grads):
        
        self._hdf5_errors = HDF5Store(name + '_errors.h5','dataset', shape_errors)
        self._hdf5_errors.write_attrs(metadata_errors)
        
        self._hdf5_grads = HDF5Store(name + '_grads.h5','dataset', shape_grads)
        self._hdf5_grads.write_attrs(metadata_grads)

        
    def write_errors(self, epoch, num_epochs, trainingLoss, validationLoss, bestLoss, bestEpoch):
        
        #self._output.write("{:3d}  {:11.4f}   {:11.4f} \n".format(epoch+1, trainingLoss, validationLoss))
        #errorsList = [epoch+1, trainingLoss, validationLoss]
                
        # if not self._errorsInit :
        #     self._errors = np.asarray(errorsList).reshape((1, len(errorsList)))
        #     self._errorsInit = True              
        # else :
        #     self._errors = np.concatenate((self._errors, np.asarray(errorsList).reshape((1, len(errorsList)))), axis=0)    
        
        self._hdf5_errors.append(np.array([epoch, trainingLoss, validationLoss, bestLoss, bestEpoch]))
        
        
    def write_grads(self, parameters):
                              
        gradNormList = []
        for p in list(filter(lambda p: p.grad is not None, parameters)):
            gradNormList.append(p.grad.data.norm(2).item())
        
        # if not self._gradNormInit :
        #     self._gradNorm = np.asarray(gradNormList).reshape((1, len(gradNormList)))
        #     self._gradNormInit = True              
        # else :
        #     self._gradNorm = np.concatenate((self._gradNorm, np.asarray(gradNormList).reshape((1, len(gradNormList)))),
        #    axis=0)
            
        self._hdf5_grads.append(np.array(gradNormList))
                                                                        
