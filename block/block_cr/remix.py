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
        self.vr_fc_loc = nn.Linear(vr_nfeat, h_size)
        self.vr_fc_jug = nn.Linear(vr_nfeat, h_size)
        self.att_loc = fusionFac([qr_size, h_size], h_size)
        self.att_fc1_loc = nn.Linear(h_size, 512)
        self.att_fc2_loc = nn.Linear(512, 1)
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
        vr_loc = vr.contiguous().view(-1, self.vr_nfeat)
        
        vr_loc = self.vr_fc_loc(vr_loc) 
        att_loc = self.att_loc([qr1, vr_loc]) 
        att_loc = self.att_fc1_loc(att_loc) 
        att_loc = F.relu(att_loc)
        att_loc = self.att_fc2_loc(att_loc).view(-1,self.vr_nbox,1) 
        att_mask_pos = nn.Sigmoid()(att_loc) 
        att_mask_neg = 1-att_mask_pos
        vr_pos_ori = att_mask_pos*vr
        vr_neg_ori = att_mask_neg*vr
        qr1_pos = qr1
        qr2_pos = qr
        vr_pos = vr_pos_ori.contiguous().view(-1, self.vr_nfeat)
        vr_pos = self.vr_fc_jug(vr_pos) 
        att_pos = self.att1_fusion([qr1_pos, vr_pos])
        att_pos = self.att1_fc1(att_pos)
        att_pos = F.relu(att_pos)
        att_pos = self.att1_fc2(att_pos).view(-1,self.vr_nbox,2)
        att_mask_position = F.softmax(att_pos, dim=1) 
        att_feature_list_pos = []
        for i in range(self.glimpse):
            t_att_mask_position = att_mask_position.narrow(2,i,1) 
            t_att_mask_position = t_att_mask_position*vr_pos_ori 
            t_att_mask_position = torch.sum(t_att_mask_position, 1, keepdim=True).squeeze() 
            att_feature_list_pos.append(t_att_mask_position)
        
        rr_pos = torch.cat(att_feature_list_pos, dim=1)
        out = self.att2_fusion([qr2_pos,rr_pos])
        qr1_neg = qr1
        qr2_neg = qr
        vr_neg = vr_neg_ori.contiguous().view(-1, self.vr_nfeat)
        vr_neg = self.vr_fc_jug(vr_neg)
        att_neg = self.att1_fusion([qr1_neg, vr_neg])
        att_neg = self.att1_fc1(att_neg)
        att_neg = F.relu(att_neg)
        att_neg = self.att1_fc2(att_neg).view(-1,self.vr_nbox,2)
        att_mask_neg = F.softmax(att_neg, dim=1) 
        att_feature_list_neg = []
        for i in range(self.glimpse):
            t_att_mask_neg = att_mask_neg.narrow(2,i,1)
            t_att_mask_neg = t_att_mask_neg*vr_neg_ori 
            t_att_mask_neg = torch.sum(t_att_mask_neg, 1, keepdim=True).squeeze()
            att_feature_list_neg.append(t_att_mask_neg)
        
        rr_neg = torch.cat(att_feature_list_neg, dim=1)
        out_neg = self.att2_fusion([qr2_neg,rr_neg])
        return out,out_neg
