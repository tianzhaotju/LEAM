import sys
import torch.utils.data as data
import pickle
import os
from vocab import VocabEntry
import numpy as np
from tqdm import tqdm
from scipy import sparse
sys.setrecursionlimit(500000000)


class SumDataset(data.Dataset):
    def __init__(self, config, dataName="train"):
        self.train_path = "train_process.txt"
        self.val_path = "dev_process.txt"
        self.test_path = "test_process.txt"
        self.Nl_Voc = {"pad": 0, "Unknown": 1}
        self.Code_Voc = {"pad": 0, "Unknown": 1}
        self.Char_Voc = {"pad": 0, "Unknown": 1}
        self.Nl_Len = config.NlLen
        self.Code_Len = config.CodeLen
        self.Char_Len = config.WoLen
        self.batch_size = config.batch_size
        self.PAD_token = 0
        self.data = None
        self.dataName = dataName
        self.Codes = []
        self.Nls = []
        self.num_step = 50
        self.ruledict = pickle.load(open("./data/rule4.pkl", "rb"))
        self.ruledict['start -> copyword2'] = len(self.ruledict)
        self.ruledict['start -> copyword3'] = len(self.ruledict)
        self.rrdict = {}
        for x in self.ruledict:
            self.rrdict[self.ruledict[x]] = x
        if not os.path.exists("./data/nl_voc.pkl"):
            self.init_dic()
        self.Load_Voc()
        if dataName == "train":
            if os.path.exists("data.pkl"):
                self.data = pickle.load(open("data.pkl", "rb"))
                return
            data = pickle.load(open('process_datacopy.pkl', 'rb'))
            print('training data length', len(data))
            self.data = self.preProcessData(data)
        elif dataName == "val":
            if os.path.exists("valdata.pkl"):
                self.data = pickle.load(open("valdata.pkl", "rb"))
                self.nl = pickle.load(open("valnl.pkl", "rb"))
                return
            self.data = self.preProcessData(open(self.val_path, "r", encoding='utf-8'))
        else:
            if os.path.exists("./data/testdata.pkl"):
                self.data = pickle.load(open("./data/testdata.pkl", "rb"))
                self.nl = pickle.load(open("./data/testnl.pkl", "rb"))
                return
            data = pickle.load(open('testcopy.pkl', 'rb'))
            self.data = self.preProcessData(data)

    def Load_Voc(self):
        if os.path.exists("./data/nl_voc.pkl"):
            self.Nl_Voc = pickle.load(open("./data/nl_voc.pkl", "rb"))
        if os.path.exists("./data/code_voc.pkl"):
            self.Code_Voc = pickle.load(open("./data/code_voc.pkl", "rb"))
            self.Code_Voc['sta'] = len(self.Code_Voc)
        if os.path.exists("./data/char_voc.pkl"):
            self.Char_Voc = pickle.load(open("./data/char_voc.pkl", "rb"))
        self.Nl_Voc["<emptynode>"] = len(self.Nl_Voc)
        self.Code_Voc["<emptynode>"] = len(self.Code_Voc)

    def init_dic(self):
        print("initVoc")
        maxCharLen = 0
        nls = []
        data = pickle.load(open('process_datacopy.pkl', 'rb'))
        for x in data:
            if len(x['rule']) > self.Code_Len:
                continue
            nls.append(x['input'])
        code_voc = VocabEntry.from_corpus(nls, size=50000, freq_cutoff=10)
        self.Code_Voc = code_voc.word2id
        for x in self.ruledict:
            lst = x.strip().lower().split()
            tmp = [lst[0]] + lst[2:]
            for y in tmp:
                if y not in self.Code_Voc:
                    self.Code_Voc[y] = len(self.Code_Voc)
        self.Nl_Voc = self.Code_Voc
        assert("root" in self.Code_Voc)
        for x in self.Nl_Voc:
            maxCharLen = max(maxCharLen, len(x))
            for c in x:
                if c not in self.Char_Voc:
                    self.Char_Voc[c] = len(self.Char_Voc)
        for x in self.Code_Voc:
            maxCharLen = max(maxCharLen, len(x))
            for c in x:
                if c not in self.Char_Voc:
                    self.Char_Voc[c] = len(self.Char_Voc)
        open("./data/nl_voc.pkl", "wb").write(pickle.dumps(self.Nl_Voc))
        open("./data/code_voc.pkl", "wb").write(pickle.dumps(self.Code_Voc))
        open("./data/char_voc.pkl", "wb").write(pickle.dumps(self.Char_Voc))

    def Get_Em(self, WordList, voc):
        ans = []
        for x in WordList:
            x = x.lower()
            if x not in voc:
                ans.append(1)
            else:
                ans.append(voc[x])
        
        return ans

    def Get_Char_Em(self, WordList):
        ans = []
        for x in WordList:
            x = x.lower()
            tmp = []
            for c in x:
                c_id = self.Char_Voc[c] if c in self.Char_Voc else 1
                tmp.append(c_id)
            ans.append(tmp)
        
        return ans

    def pad_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + [self.PAD_token] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        
        return seq

    def pad_str_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + ["<pad>"] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        
        return seq

    def pad_list(self,seq, maxlen1, maxlen2):
        if len(seq) < maxlen1:
            seq = seq + [[self.PAD_token] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        
        return seq

    def pad_multilist(self, seq, maxlen1, maxlen2, maxlen3):
        if len(seq) < maxlen1:
            seq = seq + [[[self.PAD_token] * maxlen3] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        
        return seq

    def preProcessOne(self, data):
        inputNl = []
        inputNlchar = []
        inputPos = []
        inputNlad = []
        Nl = []
        for x in data:
            inputpos = x['prob']
            tree = x['tree']
            inputpos = self.pad_seq(inputpos, self.Nl_Len)
            nl = tree.split()
            Nl.append(nl)
            node = Node('root', 0)
            currnode = node
            idx = 1
            nltmp = ['root']
            nodes = [node]
            for j, x in enumerate(nl[1:]):
                if x != "^":
                    nnode = Node(x, idx)
                    idx += 1
                    nnode.father = currnode
                    currnode.child.append(nnode)
                    currnode = nnode
                    nltmp.append(x)
                    nodes.append(nnode)
                else:
                    currnode = currnode.father
            nladrow = []
            nladcol = []
            nladdata = []
            for x in nodes:
                if x.father:
                    if x.id < self.Nl_Len and x.father.id < self.Nl_Len:
                        nladrow.append(x.id)
                        nladcol.append(x.father.id)
                        nladdata.append(1)
                    for s in x.father.child:
                        if x.id < self.Nl_Len and s.id < self.Nl_Len:
                            nladrow.append(x.id)
                            nladcol.append(s.id)
                            nladdata.append(1)
                for s in x.child:
                    if x.id < self.Nl_Len and s.id < self.Nl_Len:
                        nladrow.append(x.id)
                        nladcol.append(s.id)
                        nladdata.append(1)
            nl = nltmp
            inputnls = self.pad_seq(self.Get_Em(nl, self.Nl_Voc), self.Nl_Len)
            nlad = sparse.coo_matrix((nladdata, (nladrow, nladcol)), shape=(self.Nl_Len, self.Nl_Len))
            inputnlchar = self.Get_Char_Em(nl)
            for j in range(len(inputnlchar)):
                inputnlchar[j] = self.pad_seq(inputnlchar[j], self.Char_Len)
            inputnlchar = self.pad_list(inputnlchar, self.Nl_Len, self.Char_Len)
            inputNl.append(inputnls)
            inputNlad.append(nlad)
            inputPos.append(inputpos)
            inputNlchar.append(inputnlchar)
        self.data = [inputNl, inputNlad, inputPos, inputNlchar]
        self.nl = Nl
        
        return

    def preProcessData(self, dataFile):
        inputNl = []
        inputNlad = []
        inputNlChar = []
        inputRuleParent = []
        inputRuleChild = []
        inputParent = []
        inputParentPath = []
        inputRes = []
        inputRule = []
        inputDepth = []
        inputPos = []
        nls = []
        for i in tqdm(range(len(dataFile))):
            if len(dataFile[i]['rule']) > self.Code_Len:
                continue
            child = {}
            nl = dataFile[i]['input']
            node = Node('root', 0)
            currnode = node
            idx = 1
            nltmp = ['root']
            nodes = [node]
            for x in nl[1:]:
                if x != "^":
                    nnode = Node(x, idx)
                    idx += 1
                    nnode.father = currnode
                    currnode.child.append(nnode)
                    currnode = nnode
                    nltmp.append(x)
                    nodes.append(nnode)
                else:
                    currnode = currnode.father
            nladrow = []
            nladcol = []
            nladdata = []
            for x in nodes:
                if x.father:
                    if x.id < self.Nl_Len and x.father.id < self.Nl_Len:
                        nladrow.append(x.id)
                        nladcol.append(x.father.id)
                        nladdata.append(1)
                    for s in x.father.child:
                        if x.id < self.Nl_Len and s.id < self.Nl_Len:
                            nladrow.append(x.id)
                            nladcol.append(s.id)
                            nladdata.append(1)
                for s in x.child:
                    if x.id < self.Nl_Len and s.id < self.Nl_Len:
                        nladrow.append(x.id)
                        nladcol.append(s.id)
                        nladdata.append(1)
            nl = nltmp
            nls.append(dataFile[i]['input'])
            inputpos = dataFile[i]['problist']
            inputPos.append(self.pad_seq(inputpos, self.Nl_Len))
            inputparent = dataFile[i]['fatherlist']
            inputres = dataFile[i]['rule']
            parentname = dataFile[i]['fathername']
            for j in range(len(parentname)):
                parentname[j] = parentname[j].lower()
            inputadrow = []
            inputadcol = []
            inputaddata = []
            inputrule = [self.ruledict["start -> root"]]
            for j in range(len(inputres)):
                inputres[j] = int(inputres[j])
                inputparent[j] = int(inputparent[j]) + 1
                child.setdefault(inputparent[j], []).append(j + 1)
                if inputres[j] >= 3000000:
                    inputres[j] = len(self.ruledict) + inputres[j] - 3000000 + self.Nl_Len * 2
                    if j + 1 < self.Code_Len:
                        inputadrow.append(self.Nl_Len + j + 1)
                        inputadcol.append(inputres[j] - len(self.ruledict) - self.Nl_Len * 2)
                        inputaddata.append(1)
                    inputrule.append(self.ruledict['start -> copyword3'])
                elif inputres[j] >= 2000000:
                    inputres[j] = len(self.ruledict) + inputres[j] - 2000000
                    if j + 1 < self.Code_Len:
                        inputadrow.append(self.Nl_Len + j + 1)
                        inputadcol.append(inputres[j] - len(self.ruledict))
                        inputaddata.append(1)
                    inputrule.append(self.ruledict['start -> copyword'])
                elif inputres[j] >= 1000000:
                    inputres[j] = len(self.ruledict) + inputres[j] - 1000000 + self.Nl_Len
                    if j + 1 < self.Code_Len:
                        inputadrow.append(self.Nl_Len + j + 1)
                        inputadcol.append(inputres[j] - len(self.ruledict) - self.Nl_Len)
                        inputaddata.append(1)
                    inputrule.append(self.ruledict['start -> copyword2'])
                else:
                    inputrule.append(inputres[j])
                if inputres[j] - len(self.ruledict) >= self.Nl_Len:
                    pass
                if j + 1 < self.Code_Len:
                    inputadrow.append(self.Nl_Len + j + 1)
                    inputadcol.append(self.Nl_Len + inputparent[j])
                    inputaddata.append(1)

            inputnls = self.Get_Em(nl, self.Nl_Voc)
            inputNl.append(self.pad_seq(inputnls, self.Nl_Len))
            inputnlchar = self.Get_Char_Em(nl)
            for j in range(len(inputnlchar)):
                inputnlchar[j] = self.pad_seq(inputnlchar[j], self.Char_Len)
            inputnlchar = self.pad_list(inputnlchar, self.Nl_Len, self.Char_Len)
            inputNlChar.append(inputnlchar)
            inputruleparent = self.pad_seq(self.Get_Em(["start"] + parentname, self.Code_Voc), self.Code_Len)
            inputrulechild = []
            for x in inputrule:
                if x >= len(self.rrdict):
                    inputrulechild.append(self.pad_seq(self.Get_Em(["copyword"], self.Code_Voc), self.Char_Len))
                else:
                    rule = self.rrdict[x].strip().lower().split()
                    inputrulechild.append(self.pad_seq(self.Get_Em(rule[2:], self.Code_Voc), self.Char_Len))
            inputparentpath = []
            for j in range(len(inputres)):
                if inputres[j] in self.rrdict:
                    tmppath = [self.rrdict[inputres[j]].strip().lower().split()[0]]
                    if tmppath[0] != parentname[j].lower() and tmppath[0] == 'statements' and parentname[j].lower() == 'root':
                        tmppath[0] = 'root'
                    if tmppath[0] != parentname[j].lower() and tmppath[0] == 'start':
                        tmppath[0] = parentname[j].lower()
                    assert(tmppath[0] == parentname[j].lower())
                else:
                    tmppath = [parentname[j].lower()]
                curr = inputparent[j]
                while curr != 0:
                    if inputres[curr - 1] >= len(self.rrdict):
                        rule = 'root'
                    else:
                        rule = self.rrdict[inputres[curr - 1]].strip().lower().split()[0]
                    tmppath.append(rule)
                    curr = inputparent[curr - 1]
                inputparentpath.append(self.pad_seq(self.Get_Em(tmppath, self.Code_Voc), 10))
            inputrule = self.pad_seq(inputrule, self.Code_Len)
            inputres = self.pad_seq(inputres, self.Code_Len)
            tmp = [self.pad_seq(self.Get_Em(['start'], self.Code_Voc), 10)] + inputparentpath
            inputrulechild = self.pad_list(tmp, self.Code_Len, 10)
            inputRuleParent.append(inputruleparent)
            inputRuleChild.append(inputrulechild)
            inputRes.append(inputres)
            inputRule.append(inputrule)
            inputparent = [0] + inputparent
            inputad = sparse.coo_matrix((inputaddata, (inputadrow, inputadcol)), shape=(self.Nl_Len + self.Code_Len, self.Nl_Len + self.Code_Len))
            inputParent.append(inputad)
            inputParentPath.append(self.pad_list(inputparentpath, self.Code_Len, 10))
            nlad = sparse.coo_matrix((nladdata, (nladrow, nladcol)), shape=(self.Nl_Len, self.Nl_Len))
            inputNlad.append(nlad)
        batchs = [inputNl, inputNlad, inputRule, inputRuleParent, inputRuleChild, inputRes, inputParent, inputParentPath, inputPos, inputNlChar]
        self.data = batchs
        self.nl = nls
        if self.dataName == "train":
            open("data.pkl", "wb").write(pickle.dumps(batchs, protocol=4))
            open("nl.pkl", "wb").write(pickle.dumps(nls))
        if self.dataName == "val":
            open("valdata.pkl", "wb").write(pickle.dumps(batchs, protocol=4))
            open("valnl.pkl", "wb").write(pickle.dumps(nls))
        if self.dataName == "test":
            open("./data/testdata.pkl", "wb").write(pickle.dumps(batchs))
            open("./data/testnl.pkl", "wb").write(pickle.dumps(self.nl))
        
        return batchs

    def __getitem__(self, offset):
        ans = []
        for i in range(len(self.data)):
            d = self.data[i][offset]
            if i == 1 or i == 6:
                tmp = d.toarray().astype(np.int32)
                ans.append(tmp)
            else:
                ans.append(np.array(d))
        
        return ans

    def __len__(self):
        return len(self.data[0])


class Node:
    def __init__(self, name, s):
        self.name = name
        self.id = s
        self.father = None
        self.child = []
        self.sibiling = None
