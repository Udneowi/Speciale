# Code behind the architecture of Network 2
#
#
#
#

#Import of packages
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from torch.autograd import Variable
from datetime import date
from sklearn.metrics import roc_curve, auc
import copy
from geopy.distance import distance


#Defining the network.
class Net(nn.Module):
    
    #Defining variables upon initialization of the network
    #     input:
    #           embedding_dim        Dimension size of the embedding dimension
    #           hidden_dim           Dimension size of the hidden dimension
    #           n_loc_rank           Amount of rank location ID's 
    #           n_loc_type           Amount of type location ID's
    #           n_layers             Number of LSTM layers
    
    def __init__(self, embedding_dim, hidden_dim, n_loc_rank, n_loc_type, n_layers):
        super(Net,self).__init__()
        
        # Defining the weights of the weighted sum of the loss between exploration and location prediction
        self.weight_loss_w2v = Variable(torch.Tensor([1]).float(),requires_grad=True).cuda()
        self.weight_expl = Variable(torch.Tensor([1]).float(),requires_grad=True).cuda()
        self.weight_dist = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        
        # Defining hidden dimension
        self.hidden_dim = hidden_dim
        
        # Defining the embedding layers
        self.embeddings_rank = nn.Embedding(n_loc_rank, embedding_dim)        
        self.embeddings_type = nn.Embedding(n_loc_type, embedding_dim)        
        self.embeddings_pep  = nn.Embedding(860, embedding_dim)        
        self.embeddings_day  = nn.Embedding(8, embedding_dim)

        # Defining the network layers
        self.lstm = nn.LSTM(embedding_dim+3, hidden_dim, n_layers)
        self.decoder = nn.Linear(hidden_dim, embedding_dim)        
        self.decoder_expl = nn.Linear(hidden_dim,2)
    
    # Function which takes in the data and transforms the data into the input embeddings,
    #   and return the summation of the embeddings. 
    #   Input:
    #         x         All of the data
    #   Return:
    #         Embedding of inputs
    def embeddings_input(self, x):
        #import pdb; pdb.set_trace()
        return self.embeddings_rank(x[0].long())+self.embeddings_type(x[-1].long())+\
                self.embeddings_pep(x[5].long())+self.embeddings_day(x[3].long())
    
    # Function which takes in the data and transforms the data into the output embeddings,
    #   and return the summation of the embeddings.
    #   Input:
    #         x         All of the data
    #   Return:
    #         Embedding of output
    def embeddings_out(self, x):   
        return self.embeddings_rank(x[0].long()) + self.embeddings_type(x[-1].long())+ \
                    self.embeddings_pep(x[1].long())
    
    #def embeddings_out(self, x):   
    #    return self.embeddings_type(x[-1].long()) + self.embeddings_pep(x[1].long())
    

    # Forward step.
    #   Input:
    #         x         All of the data
    #         hidden    Weights of hidden
    #   Return:
    #         logits         logits of next location prediction
    #         hidden         new hidden tensor 
    #         logits_expl    logits of exploration prediction
    def forward(self, x, hidden):
        # Embedding of input
        embeds = self.embeddings_input(x)   # (time, bs) -> (time, bs, embedding_dim)  
        
        # Concatination of embedding and raw data that is not embedded
        embeds = torch.cat((embeds,x[[1,2,4]].permute(1,2,0).float()),2)
        
        # Actual network
        lstm_out, hidden = self.lstm(embeds,hidden)        # (time, bs, hidden_dim) -> (time, bs, hidden_dim)
        logits = self.decoder(lstm_out)                    # (time, bs, hidden_dim) -> (time, bs, n_loc)
        logits_expl = self.decoder_expl(lstm_out)
        return logits, hidden, logits_expl

# Defining the class containing the network
class Brain():
    # Initialization of variables.
    def __init__(self):
        
        # Variables used in network
        self.model = None
        self.data  = []
        self.hidden_dim = 15
        self.embedding_dim = 13
        self.n_loc_rank = 1000
        self.n_loc_type = 17040
        self.seq_len=100
        self.epochs = 100
        self.bs = 10
        self.optimizer = None
        self.lr = 0.001
        self.n_layers = 3 
        self.w_decay = 0.00001
        
        # Variables used to remember results
        self.best_model = None
        self.best_accu = 0
        self.early_stop_count = 0
        self.epoch_test = 5
        

    # Initializing the model and optimizer
    def create_model(self):
        self.model = Net(self.embedding_dim, self.hidden_dim, self.n_loc_rank,self.n_loc_type, self.n_layers)
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,weight_decay=self.w_decay)
        
        
    # Batch generator
    def batches(self, data, train = "True"):
        
        # Generating start index depending on data type
        if train == "warm":
            rand_int = data.shape[1]%self.seq_len  
        elif train == "test":
            rand_int = 0
        elif train:
            rand_int = np.random.randint(self.seq_len)  # Generates random number between 0 and seq_len.
            
        ii_starts = np.arange(rand_int, data.shape[1]-2, self.seq_len)  # Generates the array of start indices for each batch
        for ii_start in ii_starts:
            seq = data[:,ii_start: ii_start + self.seq_len+1, :]
            inp = seq[:,:-1]
            target = seq[[0,5,-1],1:].long()
            target_expl = seq[[4],1:].long()
            yield inp, target, target_expl
    
    
    # Training of the network
    def train(self, data, data_test=None,early_stop = False,data_loc=None):
        for epoch in range(self.epochs):  
            
            #Checking if training should stop early
            if early_stop:
                if self.early_stop_count == 2:
                    print("Stopped early due to overfitting")
                    break
            
            #Initialization of varibles used in results
            epoch_loss = 0
            accuracy_train = 0
            accuracy_train_expl = 0
            total_train = 0
            
            #Initialize hidden state
            hidden = self.init_hidden(self.bs)
            
            #Training on batches
            for step, (inp, target, target_expl) in enumerate(self.batches(data)):
                out, hidden, logits_expl = self.model(inp, hidden)  # Run data though network
                tmp = target.clone()                                # Cloning target tensor to construct logits
                tmp[0][target[0]>20]=20                             # Setting all location bigger than 20 to 20
                logits = torch.Tensor(inp.shape[1],self.bs,20).cuda()   # Initialize logits tensor
                
                # Create other possible targets for every given location to compute logits for them
                for i in range(20):
                    tmp[-1] = (tmp[-1]+1)%20+(tmp[-1]//20)*20  # Add 1 to the type data and keeps it in the same same interval
                    tmp_20 = tmp[0]==20                        # Remembering which locations are already 20 for the rank data 
                    tmp[0] = (tmp[0]+1)%20                     # Add 1 to the rank data mod 20
                    tmp[0][tmp_20] = 20                        # Setting those who were already 20 to 20 again
                    logits[:,:,i] = torch.sum((out*self.model.embeddings_out(tmp.clone())),dim=2)  # Compute the logits of the output
                    
                logits_dist = torch.Tensor(inp.shape[1],self.bs,20).cuda()   # Initialize logits tensor
                dat_loc_perm = data_loc[inp[5].long()].permute(3,0,1,2)
                inp_perm = inp[-3:-1].unsqueeze(3).expand_as(dat_loc_perm)
                logits_dist = torch.norm(inp_perm-dat_loc_perm,dim=0)*6371*np.pi/180  
                
                #import pdb; pdb.set_trace()
#                 for ii in range(logits_dist.shape[0]):
#                     for jj in range(logits_dist.shape[1]):
#                         for kk in range(logits_dist.shape[2]):
#                             logits_dist[ii,jj,kk] = distance(inp[-3:-1,ii,jj],data_loc[int(inp[5,ii,jj]),kk]).km
                
                logits_dist[logits_dist>100]=100
                logits_dist/=100
            
                logits_dist = self.lognorm(logits_dist)
                logits_dist[logits_dist != logits_dist]=0  # Setting NaN values to zero
                
                logits = logits + logits_dist.float()*self.model.weight_dist
                #import pdb; pdb.set_trace()
                
                # Computing the softmax of the logits
                sm = torch.softmax(logits,-1) 
                #sm_expl = torch.softmax(logits_expl,-1)
                
                target[0][target[0]>20] = 20 # Changing all target locations above 20 to 20.
                
                # Comuting the weighted loss between exploration and next location prediction
                loss = F.cross_entropy(logits.reshape(-1,20), target[0].reshape(-1),ignore_index = 20) #*self.model.weight_loss_w2v + \
                        #F.cross_entropy(logits_expl.reshape(-1,2), target_expl.reshape(-1),ignore_index = 2)*self.model.weight_expl
                
                
                
                # Saving results
                epoch_loss = epoch_loss + loss
                accuracy_train = accuracy_train + torch.sum(sm.argmax(dim=2)==target[0])
                #accuracy_train_expl += torch.sum(sm_expl.argmax(dim=2)==target_expl)
                total_train = total_train + sm.shape[0]*sm.shape[1]
                
                hidden = self.repackage_hidden(hidden) # Detaching hidden dimension so it won't be deleted
                
                # Backward step
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1) #Gradient clipping 
                self.optimizer.step()
                self.optimizer.zero_grad()
               
            #Comute results on test set and print results
            if (epoch%self.epoch_test == 0):
                print(f"Epoch: {epoch+1}/{self.epochs}", end = " | ")
                print(f"Avg epoch loss: {float(epoch_loss/(step+1)):4.3f}", end = " | ")
                print(f"Training accu: {float(accuracy_train)/total_train:4.3f}", end = " | ")
                #print(f"Training accu expl: {float(accuracy_train_expl)/total_train:4.3f}", end = " | ")
                
                #Test set
                self.test(data_test,data_loc=data_loc)
                print()
    
    #Running network on test set
    def test(self, data, pep_acc = False, data_loc = None):
            with torch.no_grad(): # No grad since it is testing
                # Saving tons of variables
                loss_vali = 0
                accuracy_vali = 0
                accuracy_vali_expl = 0
                total_vali = 0
                total_step = 0
                self.auc = {'y':[],'prob':[]}
                self.topk = {3:[],5:[]}
                pep_acc_dict = {}
                pep_guess = {}
                pred_means = []
                
                #Going through 1 user at a time
                for pep in data.keys():
                    # Initialize hidden weights and variables
                    hidden = self.init_hidden(bs=1)
                    accuracy_pep = 0
                    pep_guess[pep] = {'guess':[],'target':[]}
                    for key in self.topk.keys():
                        self.topk[key].append(0)
                    pep_acc_dict[pep] = {"correct":0,"total":0}
                    
                    #Warmup only using train data
                    for step, (inp, target, target_expl) in enumerate(self.batches(data[pep]["warm"],train="warm")):
                        _, hidden, _ = self.model(inp, hidden)
                    
                    #Actual predictions
                    for step, (inp, target, target_expl) in enumerate(self.batches(data[pep]["test"],train="test")):
                        out, hidden, logits_expl = self.model(inp, hidden) # Pass data through network
                        
                        # Creating logits vector
                        tmp = target.clone()
                        tmp[0][target[0]>20]=20
                        logits = torch.Tensor(inp.shape[1],1,20).cuda()
                        for i in range(20):
                            tmp[-1] = (tmp[-1]+1)%20+(tmp[-1]//20)*20
                            tmp_20 = tmp[0]==20
                            tmp[0] = (tmp[0]+1)%20
                            tmp[0][tmp_20] = 20
                            logits[:,:,i] = torch.sum((out*self.model.embeddings_out(tmp.clone())),dim=2)
                            
                        logits_dist = torch.Tensor(inp.shape[1],self.bs,20).cuda()   # Initialize logits tensor
                        dat_loc_perm = data_loc[inp[5].long()].permute(3,0,1,2)
                        inp_perm = inp[-3:-1].unsqueeze(3).expand_as(dat_loc_perm)
                        logits_dist = torch.norm(inp_perm-dat_loc_perm,dim=0)*6371*np.pi/180  
                        logits_dist[logits_dist>100]=100
                        logits_dist/=100
                        logits_dist = self.lognorm(logits_dist)
                        logits_dist[logits_dist != logits_dist]=0  # Setting NaN values to zero

                        logits = logits + logits_dist.float()*self.model.weight_dist
                        
                        sm = torch.softmax(logits,-1) # Softmax of logits
                        #import pdb; pdb.set_trace()
                        
                        target[0][target[0]>20] = 20 # Changing all target locations above 20 to 20.
                        
                        loss_vali += F.cross_entropy(logits.reshape(-1,20), target[0].reshape(-1),ignore_index = 20)
                        guess_print = []
                        
                        # Computing guess from softmax. However is guess is the same as current location take second choice
                        for i in range(sm.shape[0]):
                            top_guesses = torch.topk(sm[i,0,:],2)
                            if int(top_guesses.indices[0]) == int(inp[0,i]):
                                guess = top_guesses.indices[1]
                            else:
                                guess = top_guesses.indices[0]
                            
                            # Save results
                            accuracy_pep += int(target[0,i])==int(guess)
                            pep_guess[pep]['guess'].append(int(guess))
                            pep_guess[pep]['target'].append(int(target[0,i]))
                            for key in self.topk.keys():
                                self.topk[key][-1]+= int(int(target[0,i]) in torch.topk(sm[i,0,:],key)[1])
                            
                            guess_print.append(int(guess))
                            
                            pep_acc_dict[pep]['total'] += 1
                            total_vali += 1
                        
                        # Save results
                        #sm_expl = torch.softmax(logits_expl,-1)    
                        total_step += 1
                        #accuracy_vali_expl += torch.sum(sm_expl.argmax(dim=2)==target_expl)
                        #self.auc['prob'].extend(sm_expl[:,0,1].cpu().numpy())
                        #self.auc['y'].extend(target_expl.cpu().numpy().ravel())
                    
                    accuracy_vali += accuracy_pep
                    
                    for key in self.topk.keys():
                        self.topk[key][-1]/=pep_acc_dict[pep]['total']
                    
                    pep_acc_dict[pep]['correct'] += accuracy_pep
                    pred_means.append(pep_acc_dict[pep]['correct']/pep_acc_dict[pep]['total'])
                
                # Compute AUC and ROC
                #fpr, tpr, _ = roc_curve(self.auc['y'], self.auc['prob'])
                #roc_auc = auc(fpr, tpr)
                
                print(f"Test loss: {loss_vali/total_step:4.3f}", end = " | ")
                print(f"Test accu: {np.mean(pred_means):4.3f}", end = " | ")
                #print(f"Test accu expl: {float(accuracy_vali_expl)/total_vali:4.3f}", end = " | ")
                #print(f"Test expl AUC: {roc_auc:4.3f}", end = " | ")
                
                # Save best model
                if np.mean(pred_means)>self.best_accu:
                    self.best_model = copy.deepcopy(self.model.state_dict())
                    self.best_accu = np.mean(pred_means)
                    self.early_stop_count = 0
                else:
                    self.early_stop_count += 1
                #print(Counter(sm.argmax(dim=2).reshape(-1).cpu().numpy()))
                #return float(accuracy_vali)/total_vali
                if pep_acc:
                    return pep_acc_dict, pep_guess

    # Initialize hidden dimmension            
    def init_hidden(self,bs):
        weight = next(self.model.parameters())
        return (weight.new_zeros(self.n_layers, bs, self.hidden_dim),
                weight.new_zeros(self.n_layers, bs, self.hidden_dim))
    
    # Return and save best network weights.
    def best_model_dict(self, save = False):
        print(f"Returning best model with accuracy: {self.best_accu}")
        if save:
            today = date.today()
            d_today = today.strftime("%b-%d-%Y")
            torch.save(self.best_model,f"Network/network_epoch{self.epochs}_{self.best_accu:4.3f}_{d_today}")
        return self.best_model
    
    # Repackage hidden dimension to avoid deletion when doing backpropagation
    def repackage_hidden(self,h):
        """Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)
                                                                                  
    def lognorm(self, data,shape=2.292614965084133):
        return 1/(shape*data*np.sqrt(2*np.pi))*torch.exp(-torch.log(data)**2/(2*shape**2))


        