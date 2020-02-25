import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import mpmath as mp
from tqdm import tqdm
from datetime import date
import json
import copy

#Loading all the data
#  input:               Type:
#       peps            Array             An array of user id's
#       bs              Integer           Batch size, default =1
#       dat_type        String            What kind of data is loaded, "train", "test", "all". Default = "train"
#       stop_t          Integer           Stop token used inbetween path sequences. Default=925
#  output:
#       paths           PyTorch tensor    A tensor of all the attributes reshaped to the batch size  dim = [6,len(data)//bs,bs]  
def load_data(peps, bs=1,dat_type = "train",stop_t = 925):
    if stop_t == False:
        stop_token = [[],[],[],[],[],[]]
    else:    
        stop_token = [[stop_t],[2],[2],[7],[2],[855]]
    paths = np.array([[],[],[],[],[],[]])
    for pep in peps: 
        try:
            data = np.load(f"Data/{pep}/prepared_data_{dat_type}.npy")
            paths = np.concatenate((paths,stop_token,data),axis=1)
        except FileNotFoundError:
            print(f"File {pep} not found")
            continue  
    if (paths.shape[1]%bs)!=0:
        if dat_type == "test":
            paths = paths[:,:-(paths.shape[1]%bs)]
        else:
            paths = paths[:,(paths.shape[1]%bs):]
    paths = paths.reshape(len(stop_token),bs,-1)
    paths = torch.tensor(paths).permute(0,2,1)
    if torch.cuda.is_available():
        paths = paths.cuda()
    return paths

#Loading location type the data
#  input:               Type:
#       peps            Array             An array of user id's
#       bs              Integer           Batch size, default =1
#       dat_type        String            What kind of data is loaded, "train", "test", "all". Default = "train"
#       stop_t          Integer           Stop token used inbetween path sequences. Default=17035
#  output:
#       paths           PyTorch tensor    A tensor of the type locations reshaped to the batch size, dim = [1,len(data)//bs,bs]  
def load_data_W2V(peps,bs=1,dat_type = "train",stop_t = 17035):
    paths_W2V = []
    if stop_t == False:
        stop_token = []
    else:    
        stop_token = stop_t
    for i,pep in enumerate(peps):
        try:
            data = np.load(f"Data/{pep}/prepared_data_{dat_type}.npy")
            data_path = data[0,:]
            data_path[data_path>18]=19
            data_path +=20*pep  #pep eller i ?
            #import pdb; pdb.set_trace()
            paths_W2V = np.concatenate((paths_W2V,[stop_t],data_path))
        except FileNotFoundError:
            print(f"File {pep} not found")
            continue   
    paths_W2V = np.array(paths_W2V)
    
    if (len(paths_W2V)%bs)!=0:
        paths_W2V = paths_W2V[:-(len(paths_W2V)%bs)] # Makes sure the data fit our dimensions
    


    # Reshapes the list into a matrix with the batch size. 
    paths_W2V = paths_W2V.reshape(bs,-1)
    paths_W2V = torch.tensor(paths_W2V).t()
    if torch.cuda.is_available():
        paths_W2V = paths_W2V.cuda()
    return paths_W2V

    
    
# Lambda function for True entropy:
# Taken from Matlab function!
def func_Lambda(seq, i):
    n = len(seq)
    
    # Insert terminal symbol
    seq = seq + [min(seq)-1]
    
    x = 1
    mps = [idx for (idx, val) in enumerate(seq[:i]) if val == seq[i]]
    while (mps and x <= n-i):
        if mps[-1] + x >= i:
            del mps[-1]
            
        mps = [idx for idx in mps if seq[idx+x] == seq[i+x]]
        x += 1
        
    return x


# True entropy: Distribution entropy (correlaed), looks at correlations in the sequens
# S = - sum_{T'_i in T_i}(P(T'_i) * log2(P(T'_i)))
def func_S(seq,tqdm_boo = False):
    
    n = len(seq)
    L = np.zeros(n)
    
    for i in tqdm(range(n),disable = tqdm_boo):
        # Lambda function call
        L[i] = func_Lambda(seq,i)
    
    # Function from Gavin Smith (LoPpercom)
    # S = np.power(sum([L[i] / np.log2(i+1) for i in range(1,n)] ) * (1.0/n),-1)
    # Function from Morten Proschowsky (Matlab)
    S = n/sum(L) * np.log2(n)
    return S

# Predictability solving: 
# solve(0 = [-x*log2(x)-(1-x)*log2(1-x)+(1-x)*log2(N-1)] - S
# Returns value between 0 and 1.
def func_Pred(S_score,N_states):
    
    if N_states <= 1:
        return 1
    
    # Convex function for the predictibility bound
    func = lambda x, S, N: (-(x*mp.log(x,2)+(1-x)*mp.log(1-x,2))+(1-x)*mp.log(N-1,2))-S
    func2 = lambda x: func(x,S_score,N_states)
    # Solve function f(x) = 0
    res = mp.findroot(func2,0.95).real
    
    return float(res)

def entropy(seq, print_res = False,tqdm_boo = False):
    N_states = len(np.unique(seq))
    S = func_S(seq,tqdm_boo)
    U_max  = func_Pred(S,N_states)
    
    if print_res:
        print('S:      ' + str(S))
        print('U_max:  ' + str(U_max))
        
    return (S, U_max)

#Saves weights of a network with given accuracy
def save_model(state_dict,accuracy,shuffle_dict=None):
    today = date.today()
    d_today = today.strftime("%b-%d-%Y")
    ix = np.argmax(accuracy)
    torch.save(state_dict,f"Network/network_epoch_{accuracy[ix]:4.3f}_{d_today}")

    if shuffle_dict != None:
        for key, value in shuffle_dict[ix].items():
            shuffle_dict[ix][key] = int(value)
        json.dump( shuffle_dict[ix], open( f"Network/network_epoch_{accuracy[ix]:4.3f}_{d_today}_shuffle_dict.json", 'w' ) )
        
        
# Loads the training and test set given users.
def load_train_test_set(peps,bs=1):
    dat_train = load_data(peps,bs,dat_type="train_relabeled")
    dat_train_w2v = load_data_W2V(peps,bs,dat_type="train_relabeled")
    dat_train_both = torch.cat((dat_train,dat_train_w2v.reshape(1,-1,bs)),0)
    
    #Loads warms up data and actual test data
    dat_test = {}
    for pep in peps:
        dat_t = load_data([pep],bs = 1, dat_type="test_relabeled")
        dat_w = load_data([pep],bs = 1, dat_type="train_relabeled")
        dat_t_w2v = load_data_W2V([pep],bs = 1, dat_type="test_relabeled")
        dat_w_w2v = load_data_W2V([pep],bs = 1, dat_type="train_relabeled")
        dat_t_both = torch.cat((dat_t,dat_t_w2v.reshape(1,-1,1)),0)
        dat_w_both = torch.cat((dat_w,dat_w_w2v.reshape(1,-1,1)),0)
        dat_test[pep] = {"warm":dat_w_both,"test":dat_t_both}
    return dat_train_both, dat_test