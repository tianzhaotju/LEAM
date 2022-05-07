import time
from ast import arg
from torch import optim
from Dataset import SumDataset
import os
from tqdm import tqdm
from Model import *
import pickle
from ScheduledOptim import *
import sys
from Searchnode import Node
import json
import traceback
import re
from stringfycode import stringfyRoot
import javalang
import subprocess
import signal
from copy import deepcopy


def getNodeByIdS(root, ido):
    if root.id == ido:
        return root
    for x in root.child:
        t = getNodeByIdS(x, ido)
        if t:
            return t
    return None


def myHandler(signum, frame):
    raise Exception('Timeout')


def myHandler2(signum, frame):
    pass


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


args = dotdict({
    'NlLen': 500,
    'CodeLen': 60,
    'batch_size': 16,
    'embedding_size': 256,
    'WoLen': 15,
    'Vocsize': 100,
    'Nl_Vocsize': 100,
    'max_step': 3,
    'margin': 0.5,
    'poolsize': 50,
    'Code_Vocsize': 100,
    'num_steps': 50,
    'rulenum': 10,
    'cnum': 695
})
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
onelist = ['sta', 'root', 'body', 'statements', 'block', 'arguments', 'initializers', 'parameters', 'case', 'cases', 'selectors']
linenode = ['Statement_ter', 'BreakStatement_ter', 'ReturnStatement_ter', 'ContinueStatement', 'ContinueStatement_ter',
            'LocalVariableDeclaration', 'condition', 'control', 'BreakStatement', 'ContinueStatement',
            'ReturnStatement', "parameters", 'StatementExpression', 'return_type']


def getroottree2(tokens, isex=False):
    root = Node(tokens[0], 0)
    currnode = root
    idx = 1
    for x in tokens[1:]:
        if x != "^":
            nnode = Node(x, idx)
            nnode.father = currnode
            currnode.child.append(nnode)
            currnode = nnode
            idx += 1
        else:
            currnode = currnode.father
    
    return root


def save_model(model, dirs='checkpointSearch/'):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    torch.save(model.state_dict(), dirs + 'best_model.ckpt')


def load_model(model, dirs='checkpointSearch/'):
    assert os.path.exists(dirs + 'best_model.ckpt'), 'Weights for saved model not found'
    model.load_state_dict(torch.load(dirs + 'best_model.ckpt'))


def gVar(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        assert isinstance(tensor, torch.Tensor)
    if use_cuda:
        tensor = tensor.cuda()
    
    return tensor


def getAntiMask(size):
    ans = np.zeros([size, size])
    for i in range(size):
        for j in range(0, i + 1):
            ans[i, j] = 1.0
    
    return ans


def getAdMask(size):
    ans = np.zeros([size, size])
    for i in range(size - 1):
        ans[i, i + 1] = 1.0
    
    return ans


def getRulePkl(vds):
    inputruleparent = []
    inputrulechild = []
    for i in range(args.cnum):
        rule = vds.rrdict[i].strip().lower().split()
        inputrulechild.append(vds.pad_seq(vds.Get_Em(rule[2:], vds.Code_Voc), vds.Char_Len))
        inputruleparent.append(vds.Code_Voc[rule[0].lower()])
    
    return np.array(inputruleparent), np.array(inputrulechild)


def getAstPkl(vds):
    rrdict = {}
    for x in vds.Code_Voc:
        rrdict[vds.Code_Voc[x]] = x
    inputchar = []
    for i in range(len(vds.Code_Voc)):
        rule = rrdict[i].strip().lower()
        inputchar.append(vds.pad_seq(vds.Get_Char_Em([rule])[0], vds.Char_Len))
    
    return np.array(inputchar)


def evalacc(model, dev_set):
    antimask = gVar(getAntiMask(args.CodeLen))
    a, b = getRulePkl(dev_set)
    tmpast = getAstPkl(dev_set)
    tmpf = gVar(a).unsqueeze(0).repeat(4, 1).long()
    tmpc = gVar(b).unsqueeze(0).repeat(4, 1, 1).long()
    devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=args.batch_size, shuffle=False, drop_last=True,
                                            num_workers=1)
    model = model.eval()
    accs = []
    tcard = []
    loss = []
    antimask2 = antimask.unsqueeze(0).repeat(args.batch_size, 1, 1).unsqueeze(1)
    rulead = gVar(pickle.load(open("rulead.pkl", "rb"))).float().unsqueeze(0).repeat(4, 1, 1)
    tmpindex = gVar(np.arange(len(dev_set.ruledict))).unsqueeze(0).repeat(4, 1).long()
    tmpchar = gVar(tmpast).unsqueeze(0).repeat(4, 1, 1).long()
    tmpindex2 = gVar(np.arange(len(dev_set.Code_Voc))).unsqueeze(0).repeat(4, 1).long()
    for devBatch in tqdm(devloader):
        for i in range(len(devBatch)):
            devBatch[i] = gVar(devBatch[i])
        with torch.no_grad():
            l, pre = model(devBatch[0], devBatch[1], devBatch[2], devBatch[3], devBatch[4], devBatch[6], devBatch[7],
                           devBatch[8], devBatch[9], tmpf, tmpc, tmpindex, tmpchar, tmpindex2, rulead, antimask2,
                           devBatch[5])
            loss.append(l.mean().item())
            pred = pre.argmax(dim=-1)
            resmask = torch.gt(devBatch[5], 0)
            acc = (torch.eq(pred, devBatch[5]) * resmask).float()
            accsum = torch.sum(acc, dim=-1)
            resTruelen = torch.sum(resmask, dim=-1).float()
            cnum = torch.eq(accsum, resTruelen).sum().float()
            acc = acc.sum(dim=-1) / resTruelen
            accs.append(acc.mean().item())
            tcard.append(cnum.item())
    tnum = np.sum(tcard)
    acc = np.mean(accs)
    l = np.mean(loss)
    
    return acc, tnum, l


def train():
    train_set = SumDataset(args, "train")
    datalen = len(train_set.data[0])
    datalen = int(datalen * 0.9)
    rulead = gVar(pickle.load(open("rulead.pkl", "rb"))).float().unsqueeze(0).repeat(4, 1, 1)
    args.cnum = rulead.size(1)
    tmpast = getAstPkl(train_set)
    a, b = getRulePkl(train_set)
    tmpf = gVar(a).unsqueeze(0).repeat(4, 1).long()
    tmpc = gVar(b).unsqueeze(0).repeat(4, 1, 1).long()
    tmpindex = gVar(np.arange(len(train_set.ruledict))).unsqueeze(0).repeat(4, 1).long()
    tmpchar = gVar(tmpast).unsqueeze(0).repeat(4, 1, 1).long()
    tmpindex2 = gVar(np.arange(len(train_set.Code_Voc))).unsqueeze(0).repeat(4, 1).long()
    args.Code_Vocsize = len(train_set.Code_Voc)
    args.Nl_Vocsize = len(train_set.Nl_Voc)
    args.Vocsize = len(train_set.Char_Voc)
    args.rulenum = len(train_set.ruledict) + args.NlLen
    test_set = SumDataset(args, "test")
    test_set.data = [x[datalen:] for x in train_set.data]
    train_set.data = [x[:datalen] for x in train_set.data]

    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=False,
                                              drop_last=True, num_workers=1)
    model = Decoder(args)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer = ScheduledOptim(optimizer, d_model=args.embedding_size, n_warmup_steps=4000)
    maxC2 = 0
    maxL = 1e10
    if use_cuda:
        print('using GPU')
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=[0, 1])
    load_model(model)
    print('load model')
    antimask = gVar(getAntiMask(args.CodeLen))
    for epoch in range(10000):
        j = 0
        print('epoch', epoch)
        for dBatch in tqdm(data_loader):
            if j % 1000 == 10:
                acc2, tnum2, l = evalacc(model, test_set)
                print("for test " + str(acc2) + " " + str(tnum2) + " max is " + str(maxC2) + "loss is " + str(l))
                if maxL > l:
                    maxC2 = tnum2
                    maxAcc2 = acc2
                    maxL = l
                    print("find better acc " + str(maxAcc2))
                    save_model(model)
            antimask2 = antimask.unsqueeze(0).repeat(args.batch_size, 1, 1).unsqueeze(1)
            model = model.train()
            for i in range(len(dBatch)):
                dBatch[i] = gVar(dBatch[i])
            loss, _ = model(dBatch[0], dBatch[1], dBatch[2], dBatch[3], dBatch[4], dBatch[6], dBatch[7], dBatch[8],
                            dBatch[9], tmpf, tmpc, tmpindex, tmpchar, tmpindex2, rulead, antimask2, dBatch[5])
            loss = torch.mean(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()
            j += 1


class SearchNode:
    def __init__(self, ds, nl):
        self.state = [ds.ruledict["start -> root"]]
        self.prob = 0
        self.aprob = 0
        self.bprob = 0
        self.root = Node("root", 2)
        self.inputparent = ["root"]
        self.visited = {}
        self.finish = False
        self.unum = 0
        self.parent = np.zeros([args.NlLen + args.CodeLen, args.NlLen + args.CodeLen])
        self.expanded = None
        self.expandedname = []
        self.depth = [1]
        for x in ds.ruledict:
            self.expandedname.append(x.strip().split()[0])
        root = Node('root', 0)
        idx = 1
        self.idmap = {}
        self.idmap[0] = root
        currnode = root
        self.actlist = []
        for x in nl[1:]:
            if x != "^":
                nnode = Node(x, idx)
                self.idmap[idx] = nnode
                idx += 1
                nnode.father = currnode
                currnode.child.append(nnode)
                currnode = nnode
            else:
                currnode = currnode.father
        self.everTreepath = []

    def selcetNode(self, root):
        if not root.expanded and root.name in self.expandedname and root.name not in onelist:
            return root
        else:
            for x in root.child:
                ans = self.selcetNode(x)
                if ans:
                    return ans
            if root.name in onelist and root.expanded == False:
                return root
        
        return None

    def selectExpandedNode(self):
        self.expanded = self.selcetNode(self.root)

    def getRuleEmbedding(self, ds, nl):
        inputruleparent = []
        inputrulechild = []
        for x in self.state:
            if x >= len(ds.rrdict):
                inputruleparent.append(ds.Get_Em(["value"], ds.Code_Voc)[0])
                inputrulechild.append(ds.pad_seq(ds.Get_Em(["copyword"], ds.Code_Voc), ds.Char_Len))
            else:
                rule = ds.rrdict[x].strip().lower().split()
                inputruleparent.append(ds.Get_Em([rule[0]], ds.Code_Voc)[0])
                inputrulechild.append(ds.pad_seq(ds.Get_Em(rule[2:], ds.Code_Voc), ds.Char_Len))
        tmp = [ds.pad_seq(ds.Get_Em(['start'], ds.Code_Voc), 10)] + self.everTreepath
        inputrulechild = ds.pad_list(tmp, ds.Code_Len, 10)
        inputrule = ds.pad_seq(self.state, ds.Code_Len)
        inputruleparent = ds.pad_seq(inputruleparent, ds.Code_Len)
        inputdepth = ds.pad_list(self.depth, ds.Code_Len, 40)
        
        return inputrule, inputrulechild, inputruleparent, inputdepth

    def getTreePath(self, ds):
        tmppath = [self.expanded.name.lower()]
        node = self.expanded.father
        while node:
            tmppath.append(node.name.lower())
            node = node.father
        tmp = ds.pad_seq(ds.Get_Em(tmppath, ds.Code_Voc), 10)
        self.everTreepath.append(tmp)
        
        return ds.pad_list(self.everTreepath, ds.Code_Len, 10)

    def checkapply(self, rule, ds):
        if rule >= len(ds.ruledict):
            if self.expanded.name == 'root' and rule - len(ds.ruledict) >= 2 * args.NlLen:
                if len(self.expanded.child) > 0 and self.expanded.child[-1].name == 'linenode':
                    return False
                if len(self.expanded.child) == 4:
                    return False
                idx = rule - len(ds.ruledict) - 2 * args.NlLen
                if self.idmap[idx].name not in linenode:
                    return False
                if idx in self.visited:
                    return False
                return True
            if self.expanded.name == 'linenode' and rule - len(ds.ruledict) >= args.NlLen:
                if rule - len(ds.ruledict) - args.NlLen not in self.idmap:
                    return False
                if self.idmap[rule - len(ds.ruledict) - args.NlLen].name not in ['MemberReference', 'BasicType',
                                                                                 'operator', 'qualifier', 'member',
                                                                                 'Literal']:
                    return False
                if '.0' in self.idmap[rule - len(ds.ruledict) - args.NlLen].getTreestr():
                    return False
                return True
            if rule - len(ds.ruledict) >= args.NlLen:
                return False
            idx = rule - len(ds.ruledict)
            if idx not in self.idmap:
                return False
            if self.idmap[idx].name != self.expanded.name:
                if self.idmap[idx].name in ['VariableDeclarator', 'FormalParameter', 'InferredFormalParameter']:
                    return True
                return False
        else:
            rules = ds.rrdict[rule]
            if rules == 'start -> unknown':
                if self.unum >= 1:
                    return False
                return True
            if rules.strip().split()[0].lower() != self.expanded.name.lower():
                return False
        
        return True

    def copynode(self, newnode, original):
        for x in original.child:
            nnode = Node(x.name, -1)
            nnode.father = newnode
            nnode.expanded = True
            newnode.child.append(nnode)
            self.copynode(nnode, x)
        
        return

    def applyrule(self, rule, ds):
        if rule >= len(ds.ruledict):
            if rule >= len(ds.ruledict) + 2 * args.NlLen:
                idx = rule - len(ds.ruledict) - 2 * args.NlLen
            elif rule >= len(ds.ruledict) + args.NlLen:
                idx = rule - len(ds.ruledict) - args.NlLen
            else:
                idx = rule - len(ds.ruledict)
            self.actlist.append('copy-' + self.idmap[idx].name)
        else:
            self.actlist.append(ds.rrdict[rule])
        if rule >= len(ds.ruledict):
            nodesid = rule - len(ds.ruledict)
            if nodesid >= 2 * args.NlLen:
                nodesid = nodesid - 2 * args.NlLen
                nnode = Node("linenode", nodesid)
                nnode.fatherlistID = len(self.state)
                nnode.father = self.expanded
                nnode.fname = str(nodesid)
                self.expanded.child.append(nnode)
                self.visited[nodesid] = 1
            elif nodesid >= args.NlLen:
                nodesid = nodesid - args.NlLen
                nnode = Node(self.idmap[nodesid].name, nodesid)
                nnode.fatherlistID = len(self.state)
                nnode.father = self.expanded
                nnode.fname = "-" + self.printTree(self.idmap[nodesid])
                self.expanded.child.append(nnode)
            else:
                nnode = self.idmap[nodesid]
                if nnode.name == self.expanded.name:
                    self.copynode(self.expanded, nnode)
                    nnode.fatherlistID = len(self.state)
                else:
                    if nnode.name == 'VariableDeclarator':
                        currnode = -1
                        for x in nnode.child:
                            if x.name == 'name':
                                currnode = x
                                break
                        nnnode = Node(currnode.child[0].name, -1)
                    else:
                        currnode = -1
                        for x in nnode.child:
                            if x.name == 'name':
                                currnode = x
                                break
                        nnnode = Node(currnode.child[0].name, -1)
                    nnnode.father = self.expanded
                    self.expanded.child.append(nnnode)
                    nnnode.fatherlistID = len(self.state)
                self.expanded.expanded = True
        else:
            rules = ds.rrdict[rule]
            if rules == 'start -> unknown':
                self.unum += 1
            if rules.strip() == self.expanded.name + " -> End":
                self.expanded.expanded = True
            else:
                for x in rules.strip().split()[2:]:
                    nnode = Node(x, -1)
                    self.expanded.child.append(nnode)
                    nnode.father = self.expanded
                    nnode.fatherlistID = len(self.state)
        self.parent[args.NlLen + len(self.depth), args.NlLen + self.expanded.fatherlistID] = 1
        if rule >= len(ds.ruledict) + 2 * args.NlLen:
            self.parent[args.NlLen + len(self.depth), rule - len(ds.ruledict) - 2 * args.NlLen] = 1
        elif rule >= len(ds.ruledict) + args.NlLen:
            self.parent[args.NlLen + len(self.depth), rule - len(ds.ruledict) - args.NlLen] = 1
        elif rule >= len(ds.ruledict):
            self.parent[args.NlLen + len(self.depth), rule - len(ds.ruledict)] = 1
        if rule >= len(ds.ruledict) + 2 * args.NlLen:
            self.state.append(ds.ruledict['start -> copyword3'])
        elif rule >= len(ds.ruledict) + args.NlLen:
            self.state.append(ds.ruledict['start -> copyword2'])
        elif rule >= len(ds.ruledict):
            self.state.append(ds.ruledict['start -> copyword'])
        else:
            self.state.append(rule)
        self.inputparent.append(self.expanded.name.lower())
        self.depth.append(1)
        if self.expanded.name not in onelist:
            self.expanded.expanded = True
        
        return True

    def printTree(self, r):
        s = r.name + r.fname + " "
        if len(r.child) == 0:
            s += "^ "
            return s
        for c in r.child:
            s += self.printTree(c)
        s += "^ "
        return s

    def getTreestr(self):
        return self.printTree(self.root)


beamss = []


def BeamSearch(inputnl, vds, model, beamsize, batch_size, k):
    batch_size = len(inputnl[0].view(-1, args.NlLen))
    rrdic = {}
    for x in vds.Code_Voc:
        rrdic[vds.Code_Voc[x]] = x
    tmpast = getAstPkl(vds)
    a, b = getRulePkl(vds)
    tmpf = gVar(a).unsqueeze(0).repeat(2, 1).long()
    tmpc = gVar(b).unsqueeze(0).repeat(2, 1, 1).long()
    rulead = gVar(pickle.load(open("rulead.pkl", "rb"))).float().unsqueeze(0).repeat(2, 1, 1)
    tmpindex = gVar(np.arange(len(vds.ruledict))).unsqueeze(0).repeat(2, 1).long()
    tmpchar = gVar(tmpast).unsqueeze(0).repeat(2, 1, 1).long()
    tmpindex2 = gVar(np.arange(len(vds.Code_Voc))).unsqueeze(0).repeat(2, 1).long()
    with torch.no_grad():
        beams = {}
        hisTree = {}
        for i in range(batch_size):
            beams[i] = [SearchNode(vds, vds.nl[args.batch_size * k + i])]
            hisTree[i] = {}
        index = 0
        antimask = gVar(getAntiMask(args.CodeLen))
        endnum = {}
        tansV = {}
        while True:
            tmpbeam = {}
            ansV = {}
            if len(endnum) == batch_size:
                break
            if index >= args.CodeLen:
                break
            for ba in range(batch_size):
                tmprule = []
                tmprulechild = []
                tmpruleparent = []
                tmptreepath = []
                tmpAd = []
                validnum = []
                tmpdepth = []
                tmpnl = []
                tmpnlad = []
                tmpnl8 = []
                tmpnl9 = []
                for p in range(beamsize):
                    if p >= len(beams[ba]):
                        continue
                    x = beams[ba][p]
                    x.selectExpandedNode()
                    if x.expanded == None or len(x.state) >= args.CodeLen:
                        x.finish = True
                        ansV.setdefault(ba, []).append(x)
                    else:
                        validnum.append(p)
                        tmpnl.append(inputnl[0][ba].data.cpu().numpy())
                        tmpnlad.append(inputnl[1][ba].data.cpu().numpy())
                        tmpnl8.append(inputnl[8][ba].data.cpu().numpy())
                        tmpnl9.append(inputnl[9][ba].data.cpu().numpy())
                        a, b, c, d = x.getRuleEmbedding(vds, vds.nl[args.batch_size * k + ba])
                        tmprule.append(a)
                        tmprulechild.append(b)
                        tmpruleparent.append(c)
                        tmptreepath.append(x.getTreePath(vds))
                        tmpAd.append(x.parent)
                        tmpdepth.append(d)
                if len(tmprule) == 0:
                    continue
                antimasks = antimask.unsqueeze(0).repeat(len(tmprule), 1, 1).unsqueeze(1)
                tmprule = np.array(tmprule)
                tmprulechild = np.array(tmprulechild)
                tmpruleparent = np.array(tmpruleparent)
                tmptreepath = np.array(tmptreepath)
                tmpAd = np.array(tmpAd)
                tmpdepth = np.array(tmpdepth)
                tmpnl = np.array(tmpnl)
                tmpnlad = np.array(tmpnlad)
                tmpnl8 = np.array(tmpnl8)
                tmpnl9 = np.array(tmpnl9)
                result = model(gVar(tmpnl), gVar(tmpnlad), gVar(tmprule), gVar(tmpruleparent), gVar(tmprulechild),
                               gVar(tmpAd), gVar(tmptreepath), gVar(tmpnl8), gVar(tmpnl9), tmpf, tmpc, tmpindex,
                               tmpchar, tmpindex2, rulead, antimasks, None, "test")
                results = result.data.cpu().numpy()
                currIndex = 0
                for j in range(beamsize):
                    if j not in validnum:
                        continue
                    x = beams[ba][j]
                    tmpbeamsize = 0
                    result = np.negative(results[currIndex, index])
                    currIndex += 1
                    cresult = np.negative(result)
                    indexs = np.argsort(result)
                    for i in range(len(indexs)):
                        if tmpbeamsize >= 30:
                            break
                        if cresult[indexs[i]] == 0:
                            break
                        c = x.checkapply(indexs[i], vds)
                        if c:
                            tmpbeamsize += 1
                        else:
                            continue
                        prob = x.prob + np.log(cresult[indexs[i]])
                        tmpbeam.setdefault(ba, []).append([prob, indexs[i], x])
            for i in range(batch_size):
                if i in ansV:
                    if len(ansV[i]) == beamsize:
                        endnum[i] = 1
                    tansV.setdefault(i, []).extend(ansV[i])
            for j in range(batch_size):
                if j in tmpbeam:
                    if j in ansV:
                        for x in ansV[j]:
                            tmpbeam[j].append([x.prob, -1, x])
                    tmp = sorted(tmpbeam[j], key=lambda x: x[0], reverse=True)
                    beams[j] = []
                    for x in tmp:
                        if len(beams[j]) >= beamsize:
                            break
                        if x[1] != -1:
                            copynode = pickle.loads(pickle.dumps(x[2]))
                            copynode.applyrule(x[1], vds)
                            if copynode.getTreestr() in hisTree:
                                continue
                            copynode.prob = x[0]
                            beams[j].append(copynode)
                            hisTree[j][copynode.getTreestr()] = 1
                        else:
                            beams[j].append(x[2])
            index += 1
        for j in range(batch_size):
            visit = {}
            tmp = []
            for x in tansV[j]:
                if x.getTreestr() not in visit and x.finish:
                    visit[x.getTreestr()] = 1
                    tmp.append(x)
                else:
                    continue
            beams[j] = sorted(tmp, key=lambda x: x.prob, reverse=True)[:beamsize]
        return beams


def test():
    dev_set = SumDataset(args, "test")
    rulead = gVar(pickle.load(open("rulead.pkl", "rb"))).float().unsqueeze(0).repeat(2, 1, 1)
    args.cnum = rulead.size(1)
    args.Nl_Vocsize = len(dev_set.Nl_Voc)
    args.Code_Vocsize = len(dev_set.Code_Voc)
    args.Vocsize = len(dev_set.Char_Voc)
    args.rulenum = len(dev_set.ruledict) + args.NlLen
    print(dev_set.rrdict[152])
    args.batch_size = 12
    rdic = {}
    for x in dev_set.Nl_Voc:
        rdic[dev_set.Nl_Voc[x]] = x
    model = Decoder(args)
    if torch.cuda.is_available():
        print('using GPU')
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=[0, 1])
    load_model(model)
    model = model.eval()
    print('Load end.')
    
    return model


def findnodebyid(root, idx):
    if root.id == idx:
        return root
    for x in root.child:
        t = findnodebyid(x, idx)
        if t:
            return t


def getroot(strlst):
    tokens = strlst.split()
    root = Node(tokens[0], 0)
    currnode = root
    idx = 1
    for i, x in enumerate(tokens[1:]):
        if x != "^":
            nnode = Node(x, idx)
            nnode.father = currnode
            currnode.child.append(nnode)
            currnode = nnode
            idx += 1
        else:
            currnode = currnode.father
    
    return root


def getMember(node):
    for x in node.child:
        if x.name == 'member':
            return x.child[0].name


def applyoperater(ans, subroot):
    copynode = pickle.loads(pickle.dumps(subroot))
    change = False
    type = ''
    for x in ans.root.child:
        if x.id != -1:
            change = True
            node = findnodebyid(copynode, x.id)
            if node is None:
                continue
            if node.name == 'member':
                type = node.child[0].name
            elif node.name == 'MemberReference':
                type = getMember(node)
            elif node.name == 'qualifier':
                type = node.child[0].name
            elif node.name == 'operator' or node.name == 'Literal' or node.name == 'BasicType':
                type = 'valid'
            else:
                assert (0)
            idx = node.father.child.index(node)
            node.father.child[idx] = x
            x.father = node.father
    if change:
        node = Node('root', -1)
        node.child.append(copynode)
        copynode.father = node
        ans.solveroot = node
        ans.type = type
    else:
        ans.solveroot = ans.root
        ans.type = type
    
    return


def replaceVar(root, rrdict, place=False):
    if root.name in rrdict:
        root.name = rrdict[root.name]
    elif root.name == 'unknown' and place:
        root.name = "placeholder_ter"
    elif len(root.child) == 0:
        if re.match('loc%d', root.name) is not None or re.match('par%d', root.name) is not None:
            return False
    ans = True
    for x in root.child:
        ans = ans and replaceVar(x, rrdict)
    
    return ans


def getUnknown(root):
    if root.name == 'unknown':
        return [root]
    ans = []
    for x in root.child:
        ans.extend(getUnknown(x))
    
    return ans


def solveUnknown(ans, vardic, typedic, classcontent, sclassname, mode):
    nodes = getUnknown(ans.solveroot)
    fans = []
    if len(nodes) >= 2:
        return []
    elif len(nodes) == 0:
        return [ans.root.printTree(ans.solveroot)]
    else:
        unknown = nodes[0]
        if unknown.father.father and unknown.father.father.name == 'MethodInvocation':
            classname = ''
            args = []
            if unknown.father.name == 'member':
                for x in unknown.father.father.child:
                    if x.name == 'qualifier':
                        if x.child[0].name in typedic:
                            classname = typedic[x.child[0].name]
                            break
                        else:
                            if sclassname == 'org.jsoup.nodes.Element':
                                sclassname = 'org.jsoup.nodes.Node'
                            for f in classcontent[sclassname + '.java']['classes'][0]['fields']:
                                if f['name'] == x.child[0].name[:-4]:
                                    classname = f['type']
                                    break
                for x in unknown.father.father.child:
                    if x.name == 'arguments':
                        for y in x.child:
                            if y.name == 'MemberReference':
                                try:
                                    if y.child[0].child[0].name in typedic:
                                        args.append(typedic[y.child[0].child[0].name])
                                    else:
                                        args.append('int')
                                except:
                                    return []
                            elif y.name == 'Literal':
                                if y.child[0].child[0].name == "<string>_er":
                                    args.append("String")
                                else:
                                    args.append("int")
                            else:
                                return []
            if classname == '':
                classbody = classcontent[sclassname + '.java']['classes']
            elif classname != '':
                if classname + ".java" not in classcontent:
                    return []
                classbody = classcontent[classname + '.java']['classes']
            if unknown.father.name == 'qualifier':
                vtype = ""
                for x in classbody[0]['fields']:
                    if x['name'] == ans.type[:-4]:
                        vtype = x['type']
                        break
            if 'IfStatement' in ans.getTreestr():
                if mode == 1 and len(ans.solveroot.child) == 1:
                    return []
                if unknown.father.name == 'member':
                    for x in classbody[0]['methods']:
                        if len(x['params']) == 0 and x['type'] == 'boolean':
                            unknown.name = x['name'] + "_ter"
                            fans.append(unknown.printTree(ans.solveroot))
                elif unknown.father.name == 'qualifier':
                    for x in classbody[0]['fields']:
                        if x['type'] == vtype:
                            unknown.name = x['name'] + "_ter"
                            fans.append(unknown.printTree(ans.solveroot))
            else:
                if mode == 0 and ans.root == ans.solveroot and len(args) == 0 and classname != 'EndTag':
                    return []
                otype = ""
                if classname == 'EndTag':
                    otype = "String"
                if mode == 0 and ans.type != '':
                    args = []
                    if ans.type == "valid":
                        return []
                    for m in classbody[0]['methods']:
                        if m['name'] == ans.type[:-4]:
                            otype = m['type']
                            for y in m['params']:
                                args.append(y['type'])
                            break
                if unknown.father.name == 'member':
                    for x in classbody[0]['methods']:
                        if len(args) == 0 and len(x['params']) == 0:
                            if mode == 0 and x['type'] != otype:
                                continue
                            if mode == 1 and x['type'] is not None:
                                continue
                            unknown.name = x['name'] + "_ter"
                            fans.append(unknown.printTree(ans.solveroot))
                        if ans.type != '':
                            if mode == 0 and len(args) > 0 and x['type'] == otype:
                                targ = []
                                for y in x['params']:
                                    targ.append(y['type'])
                                if args == targ:
                                    unknown.name = x['name'] + "_ter"
                                    fans.append(unknown.printTree(ans.solveroot))
                        else:
                            if mode == 0 and len(args) > 0:
                                targ = []
                                for y in x['params']:
                                    targ.append(y['type'])
                                if args == targ and 'type' in x and x['type'] is None:
                                    unknown.name = x['name'] + "_ter"
                                    fans.append(unknown.printTree(ans.solveroot))
                elif unknown.father.name == 'qualifier':
                    if ans.type == 'valid':
                        return []
                    for x in classbody[0]['fields']:
                        if x['type'] == vtype:
                            unknown.name = x['name'] + "_ter"
                            fans.append(unknown.printTree(ans.solveroot))
                    for x in classbody[0]['methods']:
                        if x['type'] == vtype and len(x['params']) == 0:
                            tmpnode = Node('MethodInvocation', -1)
                            tmpnode1 = Node('member', -1)
                            tmpnode2 = Node(x['name'] + "_ter", -1)
                            tmpnode.child.append(tmpnode1)
                            tmpnode1.father = tmpnode
                            tmpnode1.child.append(tmpnode2)
                            tmpnode2.father = tmpnode1
                            unknown.name = " ".join(tmpnode.printTree(tmpnode).split()[:-1])
                            fans.append(unknown.printTree(ans.solveroot))
        elif unknown.father.name == 'qualifier':
            classbody = classcontent[sclassname + '.java']['classes']
            vtype = ""
            for x in classbody[0]['fields']:
                if x['name'] == ans.type[:-4]:
                    vtype = x['type']
                    break
            for x in classbody[0]['fields']:
                if x['type'] == vtype:
                    unknown.name = x['name'] + "_ter"
                    fans.append(unknown.printTree(ans.solveroot))
            for x in classbody[0]['methods']:
                if x['type'] == vtype and len(x['params']) == 0:
                    tmpnode = Node('MethodInvocation', -1)
                    tmpnode1 = Node('member', -1)
                    tmpnode2 = Node(x['name'] + "_ter", -1)
                    tmpnode.child.append(tmpnode1)
                    tmpnode1.father = tmpnode
                    tmpnode1.child.append(tmpnode2)
                    tmpnode2.father = tmpnode1
                    unknown.name = " ".join(tmpnode.printTree(tmpnode).split()[:-1])
                    fans.append(unknown.printTree(ans.solveroot))
        elif unknown.father.name == 'member':
            classname = ''
            if unknown.father.name == 'member':
                for x in unknown.father.father.child:
                    if x.name == 'qualifier':
                        if x.child[0].name in typedic:
                            classname = typedic[x.child[0].name]
                            break
                        else:
                            for f in classcontent[sclassname + '.java']['classes'][0]['fields']:
                                if f['name'] == x.child[0].name[:-4]:
                                    classname = f['type']
                                    break
                        if x.child[0].name[:-4] + ".java" in classcontent:
                            classname = x.child[0].name[:-4]
            if classname == '':
                classbody = classcontent[sclassname + '.java']['classes']
            elif classname != '':
                if classname + ".java" not in classcontent:
                    return []
                classbody = classcontent[classname + '.java']['classes']
            vtype = ""
            for x in classbody[0]['fields']:
                if x['name'] == ans.type[:-4]:
                    vtype = x['type']
                    break
            if unknown.father.father.father.father and (
                    unknown.father.father.father.father.name == 'MethodInvocation' or unknown.father.father.father.father.name == 'ClassCreator') and ans.type == "":
                mname = ""
                if unknown.father.father.father.father.name == "MethodInvocation":
                    tname = 'member'
                else:
                    tname = 'type'
                for s in unknown.father.father.father.father.child:
                    if s.name == 'member' and tname == 'member':
                        mname = s.child[0].name
                    if s.name == 'type' and tname == 'type':
                        mname = s.child[0].child[0].child[0].name
                idx = unknown.father.father.father.child.index(unknown.father.father)
                if tname == 'member':
                    for f in classbody[0]['methods']:
                        if f['name'] == mname[:-4] and idx < len(f['params']):
                            vtype = f['params'][idx]['type']
                            break
                else:
                    if mname[:-4] + ".java" not in classcontent:
                        return []
                    for f in classcontent[mname[:-4] + ".java"]['classes'][0]['methods']:
                        if f['name'] == mname[:-4] and idx < len(f['params']):
                            vtype = f['params'][idx]['type']
                            break
            if True:
                for x in classbody[0]['fields']:
                    if x['type'] == vtype or (x['type'] == 'double' and vtype == 'int'):
                        unknown.name = x['name'] + "_ter"
                        fans.append(unknown.printTree(ans.solveroot))
    
    return fans


def extarctmode(root):
    if len(root.child) == 0:
        return 0, None
    if root.child[0].name == 'modified':
        mode = 0
    elif root.child[0].name == 'add':
        mode = 1
    else:
        return 0, None
    root.child.pop(0)
    
    return mode, root


def getTcodes(ans, subroot, rrdict, vardic, typedic, rrdicts, classname):
    tcodes = []
    mode, ans.root = extarctmode(ans.root)
    if ans.root is None:
        return tcodes, mode
    applyoperater(ans, subroot)
    an = replaceVar(ans.solveroot, rrdict)
    if not an:
        return tcodes, mode
    try:
        tcodes = solveUnknown(ans, vardic, typedic, rrdicts, classname, mode)
    except Exception as e:
        traceback.print_exc()
        tcodes = []

    return tcodes, mode


def compileMutant(mutant, mutant_i):
    compile_savedata_one = None
    try:
        root = getroottree2(mutant['code'].split())
    except:
        return compile_savedata_one
    mode = mutant['mode']
    precode = mutant['precode']
    aftercode = mutant['aftercode']
    oldcode = mutant['oldcode']
    if '-1' in oldcode:
        return compile_savedata_one
    if mode == 1:
        aftercode = oldcode + aftercode
    lines = aftercode.splitlines()
    if len(lines) == 0:
        return compile_savedata_one
    if 'throw' in lines[0] and mode == 1:
        for s, l in enumerate(lines):
            if 'throw' in l or l.strip() == "}":
                precode += l + "\n"
            else:
                break
        aftercode = "\n".join(lines[s:])
    if lines[0].strip() == '}' and mode == 1:
        precode += lines[0] + "\n"
        aftercode = "\n".join(lines[1:])
    try:
        code = stringfyRoot(root, False, mode)
    except:
        return compile_savedata_one
    if '<string>' in code:
        if '\'.\'' in oldcode:
            code = code.replace("<string>", '"."')
        elif '\'-\'' in oldcode:
            code = code.replace("<string>", '"-"')
        elif '\"class\"' in oldcode:
            code = code.replace("<string>", '"class"')
        else:
            code = code.replace("<string>", "\"null\"")
    if len(root.child) > 0 and root.child[0].name == 'condition' and mode == 0:
        code = 'if' + code + "{"
    if code == "" and 'for' in oldcode and mode == 0:
        code = oldcode + "if(0!=1)break;"
    lnum = 0
    for l in code.splitlines():
        if l.strip() != "":
            lnum += 1
        else:
            return compile_savedata_one
    if mode == 1 and len(precode.splitlines()) > 0 and 'case' in precode.splitlines()[-1]:
        lines = precode.splitlines()
        for i in range(len(lines) - 2, 0, -1):
            if lines[i].strip() == '}':
                break
        precode = "\n".join(lines[:i])
        aftercode = "\n".join(lines[i:]) + "\n" + aftercode
    if lnum == 1 and 'if' in code and mode == 1:
        if mutant['isa']:
            code = code.replace("if", 'while')
        if len(precode.splitlines()) > 0 and 'for' in precode.splitlines()[-1]:
            code = code + 'continue;\n}\n'
        else:
            afterlines = aftercode.splitlines()
            lnum = 0
            rnum = 0
            ps = mutant
            for p, y in enumerate(afterlines):
                if ps['isa'] and y.strip() != '':
                    aftercode = "\n".join(afterlines[:p + 1] + ['}'] + afterlines[p + 1:])
                    break
                if '{' in y:
                    lnum += 1
                if '}' in y:
                    if lnum == 0:
                        aftercode = "\n".join(afterlines[:p] + ['}'] + afterlines[p:])
                        break
                    lnum -= 1
        tmpcode = precode + "\n" + code + aftercode
    else:
        tmpcode = precode + "\n" + code + aftercode
    
    return tmpcode


def containID(root):
    ans = []
    if root is None:
        return ans
    if root.position is not None:
        ans.extend([root.position.line])
    for x in root.child:
        ans.extend(containID(x))
    
    return ans


def getlineids(root, id, liness):
    tempnode = getNodeByIdS(root, id)
    cid = set(containID(tempnode))
    maxl = -1
    minl = 1e10
    for l in cid:
        maxl = max(maxl, l - 1)
        minl = min(minl, l - 1)
    
    return maxl, minl


def solveone(data, model, bugid, version):
    args.batch_size = 1
    dev_set = SumDataset(args, "test")
    dev_set.preProcessOne(data)
    indexs = 0
    devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)
    savedata = []
    if os.path.exists('%s_baseline' % (version)):
        cmd = 'rm -rf %s_baseline' % (version)
        subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1,
                         start_new_session=True).communicate()
    cmd = 'defects4j checkout -p %s -v %s%s -w %s_baseline' % (bugid.split('-')[0], bugid.split('-')[1], version[0], version)
    subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1, start_new_session=True).communicate()
    for x in tqdm(devloader):
        if indexs < 0:
            indexs += 1
            continue
        ans = BeamSearch((x[0], x[1], None, None, None, None, None, None, x[2], x[3]), dev_set, model, 64, args.batch_size, indexs)
        savedata_one = []
        for i in range(len(ans)):
            currid = indexs * args.batch_size + i
            idss = data[currid]['idss']
            if os.path.exists("result/%s.json" % idss):
                classcontent = json.load(open("result/%s.json" % idss, 'r'))
            else:
                classcontent = []
            classcontent.extend(json.load(open("temp.json", 'r')))
            rrdicts = {}
            for x in classcontent:
                rrdicts[x['filename']] = x
                if 'package_name' in x:
                    rrdicts[x['package_name'] + "." + x['filename']] = x
            vardic = data[currid]['vardic']
            typedic = data[currid]['typedic']
            classname = data[currid]['classname']
            rrdict = {}
            for x in vardic:
                rrdict[vardic[x]] = x
            for j in range(len(ans[i])):
                if j > 60 and idss != 'Lang-33':
                    break
                if len(ans[i][j].root.child) < 2:
                    continue
                precodes = []
                aftercodes = []
                oldcodes = []
                flagturn = False
                if len(ans[i][j].root.child) == 2:
                    tempnode = getNodeByIdS(data[currid]['troot'], ans[i][j].root.child[0].id)
                    cid = set(containID(tempnode))
                    maxl = -1
                    minl = 1e10
                    liness = data[currid]['code']
                    for l in cid:
                        maxl = max(maxl, l - 1)
                        minl = min(minl, l - 1)
                    if maxl == -1:
                        continue
                    precode = "\n".join(liness[0:minl])
                    aftercode = "\n".join(liness[maxl + 1:])
                    oldcode = "\n".join(liness[minl:maxl + 1])
                    precodes.append(precode)
                    aftercodes.append(aftercode)
                    oldcodes.append(oldcode)
                else:
                    maxl1, minl1 = getlineids(data[currid]['troot'], ans[i][j].root.child[0].id, data[currid]['code'])
                    maxl2, minl2 = getlineids(data[currid]['troot'], ans[i][j].root.child[2].id, data[currid]['code'])
                    if maxl1 == -1 or maxl2 == -1:
                        continue
                    liness = data[currid]['code']
                    if maxl1 < maxl2:
                        precodes.append("\n".join(liness[0:minl1]))
                        precodes.append("\n")
                        oldcodes.append("\n".join(liness[minl1:maxl1 + 1]))
                        oldcodes.append("\n".join(liness[minl2:maxl2 + 1]))
                        aftercodes.append("\n".join(liness[maxl1 + 1:minl2]))
                        aftercodes.append("\n".join(liness[maxl2 + 1:]))
                    else:
                        flagturn = True
                        precodes.append("\n")
                        print(len(liness))
                        print(minl2)
                        precodes.append("\n".join(liness[0:minl2]))
                        oldcodes.append("\n".join(liness[minl1:maxl1 + 1]))
                        oldcodes.append("\n".join(liness[minl2:maxl2 + 1]))
                        aftercodes.append("\n".join(liness[maxl1 + 1:]))
                        aftercodes.append("\n".join(liness[maxl2 + 1:minl1]))
                savedata1 = [[], []]
                for child in range(int(len(ans[i][j].root.child) / 2)):
                    tempans = deepcopy(ans[i][j])
                    tempans.root = deepcopy(ans[i][j].root.child[child * 2 + 1])
                    tempans.root.name = 'root'
                    subroot = deepcopy(ans[i][j].idmap[ans[i][j].root.child[2 * child].id])
                    tcodes, mode = getTcodes(tempans, subroot, rrdict, vardic, typedic, rrdicts, classname)
                    if flagturn:
                        idx = 1 - child
                    else:
                        idx = child
                    for code in tcodes:
                        savedata1[idx].append({
                            'id': currid,
                            'idss': idss,
                            'filename': data[currid]['filepath'],
                            'line': data[currid]['line'],
                            'mode': mode,
                            'isa': data[currid]['isa'],
                            'precode': precodes[idx],
                            'aftercode': aftercodes[idx], 
                            'oldcode': oldcodes[idx],
                            'code': code,
                            'prob': ans[i][j].prob})
                if len(savedata1[1]) == 0:
                    savedata_one.extend(savedata1[0])
                else:
                    for x in savedata1[0]:
                        for y in savedata1[1]:
                            savedata_one.append([x, y])
        for mutant_i, mutant in enumerate(savedata_one):
            if isinstance(mutant, list):
                code1 = compileMutant(mutant[0], mutant_i)
                if code1 is None:
                    continue
                code2 = compileMutant(mutant[1], mutant_i)
                if code2 is None:
                    continue
                code = code1 + code2
                mutant = mutant[0]
                mutant['code'] = code
            else:
                code = compileMutant(mutant, mutant_i)
                if code is None:
                    continue
                mutant['code'] = code
            savedata.append(mutant)
        indexs += 1
    savedata = sorted(savedata, key=lambda x: x['prob'], reverse=True)
    open('mutants/%s.json' % data[0]['idss'], 'w').write(json.dumps(savedata, indent=4))


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    if sys.argv[1] == "train":
        train()
    else:
        test()
