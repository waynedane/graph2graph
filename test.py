import dgl
import torch
import random
import glob
from dgl.nn.pytorch.conv import GATConv

import torchtext

def tokenizer(src):
    return src.split(' ')
SRC = torchtext.data.Field(tokenize=tokenizer, batch_first= True)
TRG = torchtext.data.Field(tokenize=tokenizer, batch_first= True)

dataset = torchtext.datasets.TranslationDataset(
    path='kp20k/train/train',
    exts=('.source.txt', '.target.div.txt'),
    fields=(SRC, TRG)
)

SRC.build_vocab(dataset.src)
TRG.vocab = SRC.vocab
SRC.vocab.stoi['<pad>'] =0
SRC.vocab.stoi['<unk>']=1
TRG.vocab.itos.append(';')
TRG.vocab.stoi[';'] = len(list(SRC.vocab.stoi.keys()))

print('loaded dataset!')
print('vocab size:%d'%(len(list(TRG.vocab.stoi.keys()))))
SRC.vocab.load_vectors(torchtext.vocab.FastText(language='en'))

def build_graph(instance):
    g_f = dgl.DGLGraph()
    g_b = dgl.DGLGraph()
    g_f.add_nodes(instance['node_num'])
    g_b.add_nodes(instance['node_num'])
    g_f.add_edges(instance['edges_f'][0], instance['edges_f'][1])
    g_b.add_edges(instance['edges_f'][1], instance['edges_f'][0])
    g_f.edata['e'] = torch.Tensor(instance['edata_f'])
    g_b.edata['e'] = torch.Tensor(instance['edata_f'])
    return (g_f, g_b) 
def token_to_index(source_set):
    dic = {}
    for i, token in enumerate(source_set):
        dic[token] = i

    return dic

def build_targets_graph(keyphrase_list, source_set):
    d = token_to_index(source_set)
    n = len(source_set)
    node_list = []
    for keyphrase in keyphrase_list:
        if len(keyphrase) > 1:
            for i in range(len(keyphrase)-1):
                node = [d[x] for x in  keyphrase[i:i+2]]
                node_list.append(tuple(node))
        else:
            node_list.append(tuple([d[keyphrase[0]], d[keyphrase[0]]]))
    nodelist = set(node_list)
    u=[i[0] for i in nodelist]
    v=[i[1] for i in nodelist]
  
    g = dgl.DGLGraph()
    g.add_nodes(n)
    g.add_edges(u, v)
    return g

import json
import torch.utils.data as Data
class GraphDataset(Data.Dataset):
    def __init__(self, path):
        self.path = path
        with open(self.path, 'r') as f:
            self.files = json.load(f)
    def __getitem__(self, index):
        return self.files[index]
    def __len__(self):
        return len(self.files)

datalist = glob.glob('Train_Graph/train_data*.json')

def collate_fn(instance_list):
    g_fs =[]
    g_bs =[]
    trgs = []
    bows = []
    num_nodes = []
    for instance in instance_list:
        g_f, g_b = build_graph(instance)
        target = build_targets_graph(instance['trg'], instance['bow'])
        g_fs.append(g_f)
        g_bs.append(g_b)
        trgs.append(target)
        num_nodes.append(len(instance['bow']))
        bows.extend(instance['bow'])
    return dgl.batch(g_fs),dgl.batch(g_bs), dgl.batch(trgs), torch.Tensor(bows).long(), num_nodes

import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F

class WeightedGCN(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats
                 ):
        super(WeightedGCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats)

    def forward(self,g: dgl.DGLGraph, feature: torch.Tensor):
        g.ndata['h'] = feature
        g.edata['e'] = g.edata['e']
        g.update_all(fn.src_mul_edge('h', 'e', 'm'), fn.sum('m', 'h'), self.apply_mod)
        return g.ndata.pop('h')

class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)


    def forward(self, node):
        h = self.linear(node.data['h'])
        h = F.relu(h)
        return {'h' : h}

class WordVector(nn.Module):
    def __init__(self, pre_trained_parameters):
        super(WordVector, self).__init__()
        self.embed_layer =nn.Embedding(3444546,300)
        self.embed_layer.paramters = pre_trained_parameters
        self.dropout = nn.Dropout(p=0.1, inplace=True)
    
    def forward(self, bow):
        if bow.size(0) ==1:
            bow = bow.unsqueeze(0)
        return self.dropout(self.embed_layer(bow).detach())

class IdentityLayer(nn.Module):

    def __init__(self, in_feats, out_feats):
        super(IdentityLayer, self).__init__()
        self.f = WeightedGCN(in_feats, out_feats)
        self.b = WeightedGCN(in_feats, out_feats)
        self.linear = nn.Linear(in_feats,out_feats)
        #self.dropout = nn.Dropout(p=0.5, inplace=False)

    def forward(self,g_f, g_b, feature):
        H = self.linear(feature)
        f_H = self.f(g_f, feature)
        b_H = self.b(g_b, feature)
        return H+f_H+b_H

class GATBlock(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                num_heads,
                feat_drop=0.,
                attn_drop=0.
                 ):
        super(GATBlock, self).__init__()
        self.forward_graph = GATConv(in_feats, out_feats, num_heads, feat_drop, attn_drop)
        self.backward_graph = GATConv(in_feats, out_feats, num_heads, feat_drop, attn_drop)

    def forward(self, g_f, g_b, feature):
        feature = self.forward_graph(g_f, feature) + self.backward_graph(g_b, feature)
        return feature.mean(1)
    
class ResGATBlock(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.
                 ):
        super(ResGATBlock, self).__init__()
        self.layer1 = GATBlock(in_feats, out_feats, num_heads, feat_drop, attn_drop)
        self.layer2 = GATBlock(in_feats, out_feats, num_heads, feat_drop, attn_drop)

    def forward(self, g_f, g_b, feature):
        x = self.layer1(g_f, g_b, feature)
        x = self.layer2(g_f, g_b, x)
        return x+feature

class Encoder(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 num_layers,
                 feat_drop=0.,
                 attn_drop=0.):
        super(Encoder, self).__init__()
        self.gcn_1 = IdentityLayer(in_feats, in_feats)
        self.linear_1 = nn.Linear(in_feats,out_feats)
        self.gat = nn.ModuleList([ResGATBlock(out_feats, out_feats, num_heads, feat_drop, attn_drop) for _ in range(num_layers)])
        self.norm = nn.BatchNorm1d(out_feats)

    def forward(self, g_f, g_b, feature):
        hidden = self.gcn_1(g_f, g_b, feature)
        hidden = feature+hidden*torch.sigmoid(feature)
        hidden = self.linear_1(hidden)
        for layer in self.gat:
            hidden = layer(g_f, g_b, hidden)
        return self.norm(hidden)

class Decoder(nn.Module):

    def __init__(self, alpha, diversity=False):
        super(Decoder, self).__init__()
        self.loss_func = nn.BCELoss()
        self.diversity = diversity
        self.alpha = alpha

    def forward(self, feature, targets):
        preds = self.dot_product_decode(feature)
        loss = self.compute_loss(preds, targets)
        return loss,preds

    def dot_product_decode(self, feature):
        A_pred = torch.sigmoid(torch.matmul(feature, feature.t()))
        return A_pred

    def compute_loss(self, outputs, targets):
        if self.diversity:
            loss = self.loss_func(outputs, targets) + self.alpha*torch.std(outputs, dim=-1, keepdim=False).mean()
        else:
            loss = self.loss_func(outputs, targets)
        return loss

class Graph2Graph(nn.Module):

    def __init_(self,in_feats,out_feats,num_heads,l,alpha,diversity = False,feat_drop=0.,attn_drop=0.):
        
        self.encoder = Encoder(in_feats, out_feats, num_heads, l,feat_drop, attn_drop)
        self.decoder = Decoder(alpha,diversity)

    def forward(self, g_f, g_b, feature,targets):
        feature = self.encoder(g_f, g_b, feature)
        loss = self.decoder(feature,targets)
        return loss

    def valida(self, g_f, g_b, feature):
        feature = self.encoder(g_f, g_b, feature)
        pred = self.decoder.dot_product_decode(feature)
        return pred

wordvector = WordVector(SRC.vocab.vectors)
encoder =Encoder(300,512,8,3).cuda()
decoder =Decoder(0.3,False).cuda()

import torch.optim 
optimizer = torch.optim.Adam(encoder.parameters(), lr=0.005)
encoder.train()
decoder.train()
total_loss = 0
step = 0
for epoch in range(50):
    random.shuffle(datalist)
    for datapath in datalist:
        data = GraphDataset(datapath)
        print('instance_num:%d'%(len(data)))
        dataset =Data.DataLoader(data,batch_size =32, shuffle= True, collate_fn= collate_fn)
        for idx, batch in enumerate(dataset):
            f,b,t,c, n = batch
            #feature = wordvector(c)
            feature = wordvector(c)
            
            f.to(torch.device('cuda:0'))
            b.to(torch.device('cuda:0'))
            h = encoder(f,b,feature.cuda())
            targets =  t.adjacency_matrix(transpose=True).to_dense()
            #print('check value of targets:%f'% (targets_.float().sum()))
            loss,preds = decoder(h, targets.cuda())
            #loss,preds = decoder(h, targets)
            preds_ = preds<0
            
            #print('check value of outputs:%f'% (preds_.float().sum()))
            
            print('step:%d'%(step))
            print('step_loss:%f'%(loss))
            step = step+1
            total_loss = total_loss+loss
            print('total_loss:%f'%(total_loss))
            avg_loss = total_loss/step
            print('avg_loss:%f'%(avg_loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    if epoch >1:
        torch.save({'state_dict':encoder.state_dict(),'optimizer_dict': optimizer.state_dict()},'models/model.pt')