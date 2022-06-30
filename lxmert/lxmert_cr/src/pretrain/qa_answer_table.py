# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import torch


class AnswerTable:
    ANS_CONVERT = {
        "a man": "man",
        "the man": "man",
        "a woman": "woman",
        "the woman": "woman",
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
        'grey': 'gray',
    }

    def __init__(self, dsets=None):
        self.all_ans = json.load(open("root_path/data/lxmert/all_ans.json"))
        if dsets is not None:
            dsets = set(dsets)
            # If the answer is used in the dsets
            self.anss = [ans['ans'] for ans in self.all_ans if
                         len(set(ans['dsets']) & dsets) > 0]
        else:
            self.anss = [ans['ans'] for ans in self.all_ans]
        self.ans_set = set(self.anss)

        self._id2ans_map = self.anss
        self._ans2id_map = {ans: ans_id for ans_id, ans in enumerate(self.anss)}

        assert len(self._id2ans_map) == len(self._ans2id_map)
        for ans_id, ans in enumerate(self._id2ans_map):
            assert self._ans2id_map[ans] == ans_id

    def convert_ans(self, ans):
        if len(ans) == 0:
            return ""
        ans = ans.lower()
        if ans[-1] == '.':
            ans = ans[:-1].strip()
        if ans.startswith("a "):
            ans = ans[2:].strip()
        if ans.startswith("an "):
            ans = ans[3:].strip()
        if ans.startswith("the "):
            ans = ans[4:].strip()
        if ans in self.ANS_CONVERT:
            ans = self.ANS_CONVERT[ans]
        return ans

    def ans2id(self, ans):
        return self._ans2id_map[ans]

    def id2ans(self, ans_id):
        return self._id2ans_map[ans_id]

    def ans2id_map(self):
        return self._ans2id_map.copy()

    def id2ans_map(self):
        return self._id2ans_map.copy()

    def used(self, ans):
        return ans in self.ans_set

    def all_answers(self):
        return self.anss.copy()

    @property
    def num_answers(self):
        return len(self.anss)


def load_lxmert_qa(path, model, label2ans):
    """
    Load model weights from LXMERT pre-training.
    The answers in the fine-tuned QA task (indicated by label2ans)
        would also be properly initialized with LXMERT pre-trained
        QA heads.

    :param path: Path to LXMERT snapshot.
    :param model: LXRT model instance.
    :param label2ans: The label2ans dict of fine-tuned QA datasets, like
        {0: 'cat', 1: 'dog', ...}
    :return:
    """
    print("Load QA pre-trained LXMERT from %s " % path)
    loaded_state_dict = torch.load("%s_LXRT.pth" % path)
    model_state_dict = model.state_dict()

    # Handle Multi-GPU pre-training --> Single GPU fine-tuning
    for key in list(loaded_state_dict.keys()):
        loaded_state_dict[key.replace("module.", '')] = loaded_state_dict.pop(key)

    # Isolate bert model
    '''
    bert_state_dict = {}
    for key, value in loaded_state_dict.items():
        if key.startswith('bert.'):
            bert_state_dict[key] = value
            
    '''
    #embddings
    bert_embeddings_state_dict = {}
    for key, value in loaded_state_dict.items():
        if key.startswith('bert.embeddings'):
            bert_embeddings_state_dict[key.replace("bert.embeddings.", '')] = value
    
    #visn
    bert_encoder_visn_state_dict = {}
    for key, value in loaded_state_dict.items():
        if key.startswith('bert.encoder.visn_fc'):
            bert_encoder_visn_state_dict[key.replace("bert.encoder.visn_fc.", '')] = value
    
    #layers
    bert_encoder_layer_state_dict = {}
    for key, value in loaded_state_dict.items():
        if key.startswith('bert.encoder.layer'):
            bert_encoder_layer_state_dict[key.replace("bert.encoder.layer.", '')] = value
            
    #xlayers
    bert_encoder_xlayers_state_dict = {}
    bert_encoder_xlayers_loc_state_dict = {}
    for key, value in loaded_state_dict.items():
        #Additional Key
        if key.startswith('bert.encoder.x_layers'):
            bert_encoder_xlayers_loc_state_dict[key.replace("bert.encoder.x_layers.", '')] = value
        #ini Key
        if key.startswith('bert.encoder.x_layers'):
            bert_encoder_xlayers_state_dict[key.replace("bert.encoder.x_layers.", '')] = value
        
    #rlayers
    bert_encoder_rlayers_state_dict = {}
    for key, value in loaded_state_dict.items():
        if key.startswith('bert.encoder.r_layers'):
            bert_encoder_rlayers_state_dict[key.replace("bert.encoder.r_layers.", '')] = value
    
    #bertpooler
    bert_pooler_state_dict = {}
    for key, value in loaded_state_dict.items():
        if key.startswith('bert.pooler'):
            bert_pooler_state_dict[key.replace("bert.pooler.", '')] = value
    
    # Isolate answer head
    answer_state_dict = {}
    for key, value in loaded_state_dict.items():
        if key.startswith("answer_head."):
            answer_state_dict[key.replace('answer_head.', '')] = value

    # Do surgery on answer state dict
    ans_weight = answer_state_dict['logit_fc.3.weight']
    ans_bias = answer_state_dict['logit_fc.3.bias']
    import copy
    new_answer_weight = copy.deepcopy(model_state_dict['logit_fc.3.weight'])
    new_answer_bias = copy.deepcopy(model_state_dict['logit_fc.3.bias'])
    answer_table = AnswerTable()
    loaded = 0
    unload = 0
    if type(label2ans) is list:
        label2ans = {label: ans for label, ans in enumerate(label2ans)}
    for label, ans in label2ans.items():
        new_ans = answer_table.convert_ans(ans)
        if answer_table.used(new_ans):
            ans_id_9500 = answer_table.ans2id(new_ans)
            new_answer_weight[label] = ans_weight[ans_id_9500]
            new_answer_bias[label] = ans_bias[ans_id_9500]
            loaded += 1
        else:
            new_answer_weight[label] = 0.
            new_answer_bias[label] = 0.
            unload += 1
    print("Loaded %d answers from LXRTQA pre-training and %d not" % (loaded, unload))
    print()
    answer_state_dict['logit_fc.3.weight'] = new_answer_weight
    answer_state_dict['logit_fc.3.bias'] = new_answer_bias

    # Load Bert Weights
    
    #embedding
    bert_embeddings_model_keys = set(model.lxrt_encoder.model.bert.embeddings.state_dict().keys())
    bert_embeddings_loaded_keys = set(bert_embeddings_state_dict.keys())
    assert len(bert_embeddings_model_keys - bert_embeddings_loaded_keys) == 0
    model.lxrt_encoder.model.bert.embeddings.load_state_dict(bert_embeddings_state_dict, strict=False)
    
    #visn
    bert_encoder_visn_model_keys = set(model.lxrt_encoder.model.bert.encoder.visn_fc.state_dict().keys())
    bert_encoder_visn_loaded_keys = set(bert_encoder_visn_state_dict.keys())
    assert len(bert_encoder_visn_model_keys - bert_encoder_visn_loaded_keys) == 0
    model.lxrt_encoder.model.bert.encoder.visn_fc.load_state_dict(bert_encoder_visn_state_dict, strict=False)
    
    #layers
    bert_encoder_layer_model_keys = set(model.lxrt_encoder.model.bert.encoder.layer.state_dict().keys())
    bert_encoder_layer_loaded_keys = set(bert_encoder_layer_state_dict.keys())
    assert len(bert_encoder_layer_model_keys - bert_encoder_layer_loaded_keys) == 0
    model.lxrt_encoder.model.bert.encoder.layer.load_state_dict(bert_encoder_layer_state_dict, strict=False)
    
    
    
    #xlayers_ini
    bert_encoder_xlayers_model_keys = set(model.lxrt_encoder.model.bert.encoder.x_layers.state_dict().keys())
    bert_encoder_xlayers_loaded_keys = set(bert_encoder_xlayers_state_dict.keys())
    assert len(bert_encoder_xlayers_model_keys - bert_encoder_xlayers_loaded_keys) == 0
    model.lxrt_encoder.model.bert.encoder.x_layers.load_state_dict(bert_encoder_xlayers_state_dict, strict=False)
    
    #xlayers_loc
    bert_encoder_xlayers_loc_model_keys = set(model.lxrt_encoder.model.bert.encoder.x_layers_loc.state_dict().keys())
    bert_encoder_xlayers_loc_loaded_keys = set(bert_encoder_xlayers_loc_state_dict.keys())
    assert len(bert_encoder_xlayers_loc_model_keys - bert_encoder_xlayers_loc_loaded_keys) == 0
    model.lxrt_encoder.model.bert.encoder.x_layers_loc.load_state_dict(bert_encoder_xlayers_loc_state_dict, strict=False)
    
    #rlayers
    bert_encoder_rlayers_model_keys = set(model.lxrt_encoder.model.bert.encoder.r_layers.state_dict().keys())
    bert_encoder_rlayers_loaded_keys = set(bert_encoder_rlayers_state_dict.keys())
    assert len(bert_encoder_rlayers_model_keys - bert_encoder_rlayers_loaded_keys) == 0
    model.lxrt_encoder.model.bert.encoder.r_layers.load_state_dict(bert_encoder_rlayers_state_dict, strict=False)
    
    #bertpooler
    bert_pooler_model_keys = set(model.lxrt_encoder.model.bert.pooler.state_dict().keys())
    bert_pooler_loaded_keys = set(bert_pooler_state_dict.keys())
    assert len(bert_pooler_model_keys - bert_pooler_loaded_keys) == 0
    model.lxrt_encoder.model.bert.pooler.load_state_dict(bert_pooler_state_dict, strict=False)

    # Load Answer Logic FC Weights
    model_keys = set(model.state_dict().keys())
    ans_loaded_keys = set(answer_state_dict.keys())
    assert len(ans_loaded_keys - model_keys) == 0

    model.load_state_dict(answer_state_dict, strict=False)



