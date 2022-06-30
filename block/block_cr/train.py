import os
import time
import torch
import sys
import torch.nn as nn
import utils
import random
from tqdm import tqdm
import math


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data 
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, args):
    optimizer_all = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(args)
    best_eval_score = 0
    epoch = 1
    while 1:
        total_loss = 0
        eval_loss = 0
        train_score = 0 
        eval_score = 0 
        bound = 0
        print('Epoch:', epoch, 'Start time:', time.asctime(time.localtime(time.time())))
        iter_wrapper = lambda x: tqdm(x, total=len(train_loader), ncols=60)
        t = time.time()
        for i, (in_v, in_b, in_q, in_qlen, out_a) in iter_wrapper(enumerate(train_loader)):
            in_v = in_v.cuda()
            in_q = in_q.cuda()
            in_qlen = in_qlen.cuda() 
            out_a = out_a.cuda()
            pred, pred_neg = model(in_v, in_q, in_qlen)
            target_flat = pred_neg.clone().detach()
            target_flat = nn.Softmax(-1)(target_flat)
            target_flat = target_flat-out_a
            target_flat = target_flat.clamp_(0,1)
            target_neg = pred_neg.clone().detach().fill_(0)
            target_neg[:,0] = 1
            loss = instance_bce_with_logits(pred, out_a) + \
                    args.ls_neg*instance_bce_with_logits(pred_neg, target_neg) + \
                    args.ls_flat*instance_bce_with_logits(pred_neg, target_flat)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_all.step()
            optimizer_all.zero_grad()

            batch_score = compute_score_with_logits(pred, out_a.data).sum().item()
            total_loss += loss.item() * args.batch_size
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        
        if None != eval_loader:
            model.train(False)
            eval_loss, eval_score, bound = evaluate(model, eval_loader, args)
            model.train(True)
        logger.log([epoch, total_loss, train_score, eval_loss, 100*eval_score, 100*bound, time.time()-t])
        if eval_score > best_eval_score :
            model_path = './save/' + args.cp_name + '_model.pth'
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score
        if None == eval_loader:
            model_path = './save/' + args.cp_name + '_e' + str(epoch) + '.pth'
            torch.save(model.state_dict(), model_path)
        epoch = epoch+1
        if epoch> args.max_epoch and args.max_epoch>0:
            break


def evaluate(model, dataloader, args):
    total_loss = 0
    score = 0
    upper_bound = 0
    num_data = 0
    for in_v, in_b, in_q, in_qlen, out_a in iter(dataloader):
        in_v = in_v.cuda()
        in_q = in_q.cuda()
        in_qlen = in_qlen.cuda()
        out_a = out_a.cuda()
        with torch.no_grad():
            pred,pred_neg = model(in_v, in_q, in_qlen)
            loss = instance_bce_with_logits(pred, out_a)
        batch_score = compute_score_with_logits(pred, out_a).sum().item()
        score += batch_score
        total_loss += loss.item() * args.batch_size
        upper_bound += (out_a.max(1)[0]).sum().item()
        num_data += pred.size(0)

    total_loss = total_loss / len(dataloader.dataset)
    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return total_loss, score, upper_bound
