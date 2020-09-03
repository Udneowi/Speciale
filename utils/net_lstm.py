import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, n_loc, n_layers):
        super(Net,self).__init__()
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(n_loc, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim+4, hidden_dim, n_layers)

        self.decoder = nn.Linear(hidden_dim, n_loc)

    def forward(self, x, hidden):
        embeds = self.embeddings(x[0].long())                       # (time, bs) -> (time, bs, embedding_dim)
        embeds = torch.cat((embeds,x[1:5].permute(1,2,0).float()),2)
        lstm_out, hidden = self.lstm(embeds,hidden)                  # (time, bs, hidden_dim) -> (time, bs, hidden_dim)
        logits = self.decoder(lstm_out)                    # (time, bs, hidden_dim) -> (time, bs, n_loc)
        return logits, hidden
    
class Brain():
    def __init__(self):
        self.model = None
        self.data  = []
        self.hidden_dim = 13
        self.embedding_dim = 15
        self.n_loc = 1000
        self.seq_len=30
        self.epochs = 100
        self.bs = 10
        self.optimizer = None
        self.lr = 0.001
        self.n_layers = 3
        self.w_decay = 0.00001

        
    def create_model(self):
        self.model = Net(self.embedding_dim, self.hidden_dim, self.n_loc, self.n_layers)
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,weight_decay=self.w_decay)
        
    def batches(self, data, train = "True"):
        if train == "warm":
            rand_int = data.shape[1]%self.seq_len  # Generates random number between 0 and seq_len.
        elif train == "test":
            rand_int = 0
        elif train:
            rand_int = np.random.randint(self.seq_len)  # Generates random number between 0 and seq_len.
            
        ii_starts = np.arange(rand_int, data.shape[1] - 2, self.seq_len)  # Generates the array of start indices for each batch
        for ii_start in ii_starts:
            seq = data[:,ii_start: ii_start + self.seq_len+1, :]
            inp = seq[:,:-1]
            target = seq[0,1:].long()
            yield inp, target
    
    def train(self, data, data_test=None):
        for epoch in range(self.epochs):
            #print(f"Epoch: {epoch+1}/{self.epochs}")
            epoch_loss = 0
            accuracy_train = 0
            total_train = 0
            hidden = self.init_hidden(self.bs)
            for step, (inp, target) in enumerate(self.batches(data)):
                
                logits, hidden = self.model(inp, hidden)
                sm = torch.softmax(logits,-1)
                loss = F.cross_entropy(logits.reshape(-1,self.n_loc), target.reshape(-1))
                epoch_loss += loss
                accuracy_train +=  torch.sum(target==sm.argmax(dim=2))
                
                total_train += sm.shape[0]*sm.shape[1]
                hidden = self.repackage_hidden(hidden)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if (epoch%10 == 0):
                print(f"Epoch: {epoch+1}/{self.epochs}", end = " | ")
                print(f"Avg epoch loss: {epoch_loss/(step+1):4.3f}", end = " | ")
                print(f"Training accu: {float(accuracy_train)/total_train:4.3f}", end = " | ")
                if (data_test != None):
                    self.test(data_test)
                print()
            
    def test(self, data, pep_acc = False):
            with torch.no_grad():
                loss_vali = 0
                accuracy_vali = 0
                accuracy_vali_old = 0
                total_vali = 0
                total_step = 0
                self.topk = {3:[],5:[]}
                
                pep_acc_dict = {}
                for pep in data.keys():
                    hidden = self.init_hidden(bs=1)
                    accuracy_pep = 0
                    for key in self.topk.keys():
                        self.topk[key].append(0)
                
                    pep_acc_dict[pep] = {"correct":0,"total":0}
                    for step, (inp, target) in enumerate(self.batches(data[pep]["warm"],train="warm")):
                        _, hidden = self.model(inp, hidden)
                    for step, (inp, target) in enumerate(self.batches(data[pep]["test"],train="test")):
                        logits, hidden = self.model(inp, hidden)
                        sm = torch.softmax(logits,-1)
                        loss_vali += F.cross_entropy(logits.reshape(-1,self.n_loc), target.reshape(-1))
                        
                        for i in range(sm.shape[0]):
                            top_guesses = torch.topk(sm[i,0,:],2)
                            if int(top_guesses.indices[0]) == int(inp[0,i]):
                                guess = top_guesses.indices[1]
                            else:
                                guess = top_guesses.indices[0]
                            accuracy_pep += int(target[i])==int(guess)
                        
                            pep_acc_dict[pep]['total'] += 1
                               
                            for key in self.topk.keys():
                                self.topk[key][-1]+= int(int(target[i]) in torch.topk(sm[i,0,:],key)[1])
                        
    
                        accuracy_vali_old +=  torch.sum(target==sm.argmax(dim=2))
                        
                        total_vali += sm.shape[0]*sm.shape[1]
                        total_step += 1
                    accuracy_vali += accuracy_pep  
                    for key in self.topk.keys():
                        self.topk[key][-1]/=pep_acc_dict[pep]['total']
                    
                    pep_acc_dict[pep]['correct'] += accuracy_pep
                print(f"Test loss: {loss_vali/total_step:4.3f}", end = " | ")
                print(f"Test accu: {float(accuracy_vali)/total_vali:4.3f}", end = " | ")
                print(f"Test accu old: {float(accuracy_vali_old)/total_vali:4.3f}", end = " | ")
                if pep_acc:
                    return pep_acc_dict

                
    def init_hidden(self,bs):
        weight = next(self.model.parameters())
        return (weight.new_zeros(self.n_layers, bs, self.hidden_dim),
                weight.new_zeros(self.n_layers, bs, self.hidden_dim))
    
    def repackage_hidden(self,h):
        """Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)
        