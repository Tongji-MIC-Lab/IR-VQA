import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from fusions import LinearSum, ConcatMLP, MLB, MFB, MFH, Block


class nn_qAtt_cf(nn.Module):
    def __init__(self, qr_len, qr_size, h_size):
        super(nn_qAtt_cf, self).__init__()
        self.q_att_fc1 = nn.Linear(qr_size, 512)
        self.q_att_fc2 = nn.Linear(512, 2)
        self.qr_len = qr_len
        self.qr_size = qr_size
        
    def mask_softmax(self, x, lengths):
        mask = torch.ones_like(x).to(device=x.device, non_blocking=True)
        for b in range(x.size(0)):
            mask[b,lengths[b]:,:] = 0
        x = torch.exp(x)
        x = x * mask
        x = x / torch.sum(x, dim=1, keepdim=True).expand_as(x)
        return x  
    
    def forward(self, qr, qlens): 
        qr_att = qr.contiguous().view(-1, self.qr_size)
        qr_att = self.q_att_fc1(qr_att)
        qr_att = F.relu(qr_att)
        qr_att = self.q_att_fc2(qr_att)
        qr_att = qr_att.contiguous().view(-1,self.qr_len,2)
        qr_att = self.mask_softmax(qr_att, qlens)
        if qr_att.size(2) > 1:
            qr_atts = torch.unbind(qr_att, dim=2)
            qr_outs = []
            for qr_att in qr_atts:
                qr_att = qr_att.unsqueeze(2)
                qr_att = qr_att.expand_as(qr)
                qr_out = qr_att*qr
                qr_out = qr_out.sum(1)
                qr_outs.append(qr_out)
            qr = torch.cat(qr_outs, dim=1)
        return qr


class nn_reasoning_block(nn.Module):
    def __init__(self, qr_len, qr_size, vr_nbox, vr_nfeat, h_size, o_size):
        super(nn_reasoning_block, self).__init__()
        self.glimpse = 2
        #
        methodName = 'Block'
        if methodName == 'MLB':
            fusionFac = MLB
        elif methodName == 'MFH':
            fusionFac = MFH
        elif methodName == 'Block':
            fusionFac = Block
        else:
            print('error occurs for lost the fusion name')
            sys.exit(1)
        self.att_qr = nn_qAtt_cf(qr_len, qr_size, h_size)
        qr_size = qr_size*2
        #qv_att
        self.v1_fc = nn.Linear(vr_nfeat, h_size)
        self.att1_fusion = fusionFac([qr_size, h_size], h_size)
        self.att1_fc1 = nn.Linear(h_size, 512)
        self.att1_fc2 = nn.Linear(512, self.glimpse)
        
        self.att2_fusion = fusionFac([qr_size, vr_nfeat*self.glimpse], o_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.qr_len = qr_len
        self.qr_size = qr_size
        self.vr_nbox = vr_nbox
        self.vr_nfeat = vr_nfeat
        self.h_size = h_size
        self.o_size = o_size
        

    def forward(self, qr, vr, qlens):
        qr = self.att_qr(qr, qlens)
        qr1 = qr
        qr1 = qr1.view(-1,1,qr1.size(1)) 
        qr1 = qr1.repeat(1,self.vr_nbox,1).view(-1,qr1.size(2)) 
        vr1 = vr.contiguous().view(-1, self.vr_nfeat)
        vr1 = self.v1_fc(vr1)
        
        att1 = self.att1_fusion([qr1, vr1]) 
        att1 = self.att1_fc1(att1) 
        att1 = F.relu(att1)
        att1 = self.att1_fc2(att1).view(-1,self.vr_nbox,2) 
        att1_mask = F.softmax(att1, dim=1)
        att1_feature_list = []
        for i in range(self.glimpse):
            t_att1_mask = att1_mask.narrow(2,i,1) 
            t_att1_mask = t_att1_mask*vr 
            t_att1_mask = torch.sum(t_att1_mask, 1, keepdim=True).squeeze()
            att1_feature_list.append(t_att1_mask)
        
        vr2 = torch.cat(att1_feature_list, dim=1) 
        qr2 = qr
        out = self.att2_fusion([qr2,vr2])
        return out
