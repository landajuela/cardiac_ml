import torch.nn as nn

# Initialitation
def init_weights(self):
    for idx, m in enumerate(self.modules()):         
        if idx > 0:
            #print('{} -> {}'.format(idx,m))    
            if type(m) in [nn.RNN, nn.GRU, nn.LSTM, nn.RNNCell, nn.GRUCell, nn.LSTMCell]:
                for name, param in m.named_parameters():
                    print(f'Initialization of {name}', end="", flush=True)
                    if 'weight_ih' in name:   
                        nn.init.xavier_uniform_(param.data)
                        print('...done')
                    elif 'weight_hh' in name:
                        nn.init.xavier_uniform_(param.data)
                        print('...done')
                    elif 'bias' in name:
                        param.data.fill_(0)
                        print('...done')
            if type(m) in [nn.Linear]:
                for name, param in m.named_parameters():
                    print(f'Initialization of {name}', end="", flush=True)
                    if 'weight' in name:   
                        nn.init.xavier_uniform_(param.data)
                        print('...done')
                    elif 'bias' in name:
                        param.data.fill_(0)
                        print('...done')

def get_name_parameters(model):
    names_of_parameters = [] 
    for idx, m in enumerate(model.modules()):
        if idx > 0:
            if type(m) is not nn.Sequential:
                for name, param in m.named_parameters():
                    if param.requires_grad:
                        names_of_parameters.append(str(idx-1) + '_' + name)               
    return names_of_parameters
