import sys
sys.path.append('./')
sys.path.append('./model')
import time
import javalang
import subprocess
from model.run import *
from model.Searchnode import Node


linenode = ['Statement_ter', 'BreakStatement_ter', 'ReturnStatement_ter', 'ContinueStatement', 'ContinueStatement_ter', 'LocalVariableDeclaration', 'condition', 'control', 'BreakStatement', 'ContinueStatement', 'ReturnStatement', "parameters", 'StatementExpression', 'return_type']


def getLocVar(node):
  varnames = []
  if node.name == 'VariableDeclarator':
    currnode = -1
    for x in node.child:
      if x.name == 'name':
        currnode = x
        break
    varnames.append((currnode.child[0].name, node))
  if node.name == 'FormalParameter':
    currnode = -1
    for x in node.child:
      if x.name == 'name':
        currnode = x
        break
    varnames.append((currnode.child[0].name, node))
  if node.name == 'InferredFormalParameter':
    currnode = -1
    for x in node.child:
      if x.name == 'name':
        currnode = x
        break
    varnames.append((currnode.child[0].name, node))
  for x in node.child:
    varnames.extend(getLocVar(x))
  return varnames


n = 0
def setid(root):
  global n
  root.id = n
  n += 1
  for x in root.child:
    setid(x)


def solveLongTree(root, subroot):
    global n
    m = 'None'
    troot = 'None'
    for x in root.child:
        if x.name == 'name':
            m = x.child[0].name
    if len(root.getTreestr().strip().split()) >= 1000:
        tmp = subroot
        if len(tmp.getTreestr().split()) >= 1000:
            assert(0)
        lasttmp = None
        while True:
            if len(tmp.getTreestr().split()) >= 1000:
                break
            lasttmp = tmp
            tmp = tmp.father
        index = tmp.child.index(lasttmp)
        ansroot = Node(tmp.name, 0)
        ansroot.child.append(lasttmp)
        ansroot.num = 2 + len(lasttmp.getTreestr().strip().split())
        while True:
            b = True
            afternode = tmp.child.index(ansroot.child[-1]) + 1
            if afternode < len(tmp.child) and ansroot.num + tmp.child[afternode].getNum() < 1000:
                b = False
                ansroot.child.append(tmp.child[afternode])
                ansroot.num += tmp.child[afternode].getNum()
            prenode = tmp.child.index(ansroot.child[0]) - 1
            if prenode >= 0 and ansroot.num + tmp.child[prenode].getNum() < 1000:
                b = False
                ansroot.child.append(tmp.child[prenode])
                ansroot.num += tmp.child[prenode].getNum()
            if b:
                break
        troot = ansroot
    else:
        troot = root
    n = 0
    setid(troot)
    varnames = getLocVar(troot)
    fnum = -1
    vnum = -1
    vardic = {}
    vardic[m] = 'meth0'
    typedic = {}
    for x in varnames:
        if x[1].name == 'VariableDeclarator':
            vnum += 1
            vardic[x[0]] = 'loc' + str(vnum)
            t = -1
            for s in x[1].father.father.child:
                if s.name == 'type' and t == -1:
                    if len(s.child[0].child) > 1:
                        try:
                            t = s.child[0].child[1].child[0].child[0].child[0].name[:-4]
                        except:
                            t = s.child[0].child[0].child[0].name[:-4]
                    else:
                        t = s.child[0].child[0].child[0].name[:-4]
            assert(t != -1)
            typedic[x[0]] = t
        else:
            fnum += 1
            vardic[x[0]] = 'par' + str(fnum)
            t = -1
            for s in x[1].child:
                if s.name == 'type' and t == -1:
                    if len(s.child[0].child) > 1:
                        try:
                            t = s.child[0].child[1].child[0].child[0].child[0].name[:-4]
                        except:
                            t = s.child[0].child[0].child[0].name[:-4]
                    else:
                        t = s.child[0].child[0].child[0].name[:-4]
                    break
            assert(t != -1)
            typedic[x[0]] = t
    return troot, vardic, typedic


def addter(root):
    if len(root.child) == 0:
        root.name += "_ter"
    for x in root.child:
        addter(x)
    return


def setProb(r, p):
    r.possibility =  p
    for x in r.child:
        setProb(x, p)


def getLineNode(root, block, add=True):
  ans = []
  block = block + root.name
  for x in root.child:
    if x.name in linenode:
      if 'info' in x.getTreestr() or 'assert' in x.getTreestr() or 'logger' in x.getTreestr() or 'LOGGER' in x.getTreestr() or 'system.out' in x.getTreestr().lower():
        continue
      x.block = block
      ans.append(x)
    else:
      if not add:
        s = block
      else:
        s = block + root.name
      tmp = getLineNode(x, block)
      ans.extend(tmp)
  return ans


def getroottree(tokens, isex=False):
    root = Node(tokens[0], 0)
    currnode = root
    idx = 1
    for i, x in enumerate(tokens[1:]):
        if x != "^":
            if isinstance(x, tuple):
                nnode = Node(x[0], idx)
                nnode.position = x[1]
            else:
                nnode = Node(x, idx)
            nnode.father = currnode
            currnode.child.append(nnode)
            currnode = nnode
            idx += 1
        else:
            currnode = currnode.father
    return root


def ismatch(root, subroot):
    index = 0
    for x in subroot.child:
        while index < len(root.child) and root.child[index].name != x.name:
            index += 1
        if index == len(root.child):
            return False
        if not ismatch(root.child[index], x):
            return False
        index += 1
    return True


def findSubtree(root, subroot):
    if root.name == subroot.name:
        if ismatch(root, subroot):
            return root
    for x in root.child:
        tmp = findSubtree(x, subroot)
        if tmp:
            return tmp
    return None


def generateAST(tree):
    sub = []
    if not tree:
        return ['None', '^']
    if isinstance(tree, str):
        tmpStr = tree
        tmpStr = tmpStr.replace(" ", "").replace(":", "")
        if "\t" in tmpStr or "'" in tmpStr or "\"" in tmpStr:
            tmpStr = "<string>"
        if len(tmpStr) == 0:
            tmpStr = "<empty>"
        if tmpStr[-1] == "^":
            tmpStr += "<>"
        sub.append(tmpStr)
        sub.append("^")
        return sub
    if isinstance(tree, list):
        if len(tree) == 0:
            sub.append("empty")
            sub.append("^")
        else:
            for ch in tree:
                subtree = generateAST(ch)
                sub.extend(subtree)
        return sub
    position = None
    if hasattr(tree, 'position'):
        position = tree.position
    curr = type(tree).__name__
    sub.append((curr, position))
    try:
        for x in tree.attrs:
            if x == "documentation":
                continue
            if not getattr(tree, x):
                continue
            sub.append(x)
            node = getattr(tree, x)
            if isinstance(node, list):
                if len(node) == 0:
                    sub.append("empty")
                    sub.append("^")
                else:
                    for ch in node:
                        subtree = generateAST(ch)
                        sub.extend(subtree)
            elif isinstance(node, javalang.tree.Node):
                subtree = generateAST(node)
                sub.extend(subtree)
            elif not node:
                continue
            elif isinstance(node, str):
                tmpStr = node
                tmpStr = tmpStr.replace(" ", "").replace(":", "")
                if "\t" in tmpStr or "'" in tmpStr or "\"" in tmpStr:
                    tmpStr = "<string>"
                if len(tmpStr) == 0:
                    tmpStr = "<empty>"
                if tmpStr[-1] == "^":
                    tmpStr += "<>"
                sub.append(tmpStr)
                sub.append("^")
            elif isinstance(node, set):
                for ch in node:
                    subtree = generateAST(ch)
                    sub.extend(subtree)
            elif isinstance(node, bool):
                sub.append(str(node))
                sub.append("^")
            else:
                assert(0)
            sub.append("^")
    except AttributeError:
        assert(0)
        pass
    sub.append('^')
    return sub


def getSubroot(treeroot):
    currnode = treeroot
    lnode = None
    mnode = None
    while currnode:
        if currnode.name in linenode:
            lnode = currnode
            break
        currnode = currnode.father
    currnode = treeroot
    while currnode:
        if currnode.name == 'MethodDeclaration' or currnode.name == 'ConstructorDeclaration':
            mnode = currnode
            break
        currnode = currnode.father
    return lnode, mnode


def getNodeById(root, line):
    if root.position:
        if root.position.line == line and root.name != 'IfStatement' and root.name != 'ForStatement':
            return root
    for x in root.child:
        t = getNodeById(x, line)
        if t:
            return t
    return None


def containID(root):
    ans = []
    if root.position is not None:
        ans.extend([root.position.line])
    for x in root.child:
        ans.extend(containID(x))
    return ans


def getAssignMent(root):
    if root.name == 'Assignment':
        return root
    for x in root.child:
        t = getAssignMent(x)
        if t:
            return t
    return None


def isAssign(line):
    if 'Assignment' not in line.getTreestr():
        return False
    anode = getAssignMent(line)
    if anode.child[0].child[0].name == 'MemberReference' and anode.child[1].child[0].name == 'MethodInvocation':
        try:
            m = anode.child[0].child[0].child[0].child[0].name
            v = anode.child[1].child[0].child[0].child[0].name
        except:
            return False
        return m == v
    if anode.child[0].child[0].name == 'MemberReference':
        try:
            m = anode.child[0].child[0].child[0].child[0].name
        except:
            return False
        if "qualifier " + m in anode.child[1].getTreestr():
            return True
    return False


lst = ['Chart-1', 'Chart-8', 'Chart-9', 'Chart-11', 'Chart-12', 'Chart-20', 'Chart-24', 'Chart-26', 'Closure-14', 'Closure-15', 'Closure-62', 'Closure-63', 'Closure-73', 'Closure-86', 'Closure-92', 'Closure-93', 'Closure-104', 'Closure-118', 'Closure-124', 'Lang-6', 'Lang-26', 'Lang-33', 'Lang-38', 'Lang-43', 'Lang-45', 'Lang-51', 'Lang-55', 'Lang-57', 'Lang-59', 'Math-5', 'Math-27', 'Math-30', 'Math-33', 'Math-34', 'Math-41', 'Math-50', 'Math-57', 'Math-59', 'Math-70', 'Math-75', 'Math-80', 'Math-94', 'Math-105', 'Time-4', 'Time-7']
model = test()
bugid = sys.argv[1]
prlist = [bugid.split("-")[0]]
ids = [[int(bugid.split("-")[1])]]
for i, xss in enumerate(prlist):
    for idx in ids[i]:
        idss = xss + "-" + str(idx)
        if idss != bugid:
            continue
        timecurr = time.time()
        x = xss
        locationdir = 'location2/%s/%d/parsed_ochiai_result' % (x, idx)
        methodvisit = {}
        if not os.path.exists(locationdir):
            continue
        if os.path.exists('fixed'):
            cmd = 'rm -rf fixed'
            subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1, start_new_session=True).communicate()
        cmd = 'defects4j checkout -p %s -v %sf -w fixed' % (bugid.split('-')[0], bugid.split('-')[1])
        subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1, start_new_session=True).communicate()

        lines = open(locationdir, 'r').readlines()
        location = []
        locationdict = {}
        for loc in lines:
            lst = loc.strip().split()
            prob = eval(lst[1])
            classname, lineid= lst[0].split('#')
            location.append((classname, prob, eval(lineid)))
            locationdict[lst[0]] = (classname, prob, eval(lineid))         
        dirs = os.popen('defects4j export -p dir.src.classes -w fixed').readlines()[-1]

        data = []
        for j in range(len(location)):
            ac = location[j]
            classname = ac[0]
            if '$' in classname:
                classname = classname[:classname.index('$')]
            s = classname
            filepath = "fixed/%s/%s.java" % (dirs, s.replace('.', '/'))
            lines1 = open(filepath, "r").read().strip()
            liness = lines1.splitlines()
            tokens = javalang.tokenizer.tokenize(lines1)
            parser = javalang.parser.Parser(tokens)
            tree = parser.parse()
            tmproot = getroottree(generateAST(tree))
            lineid = ac[2]

            currroot = getNodeById(tmproot, lineid)
            lnode, mnode = getSubroot(currroot)
            if mnode is None:
                continue
            tree = mnode.printTree(mnode)
            if tree not in methodvisit:
                methodvisit[tree] = 1
            else:
                continue
            oldcode = liness[ac[2] - 1]
            isIf = True
            subroot = lnode
            treeroot = mnode
            presubroot = None
            aftersubroot = None
            linenodes = getLineNode(treeroot, "")
            if subroot not in linenodes:
                continue
            currid = linenodes.index(subroot)
            if currid > 0:
                presubroot = linenodes[currid - 1]
            if currid < len(linenodes) - 1:
                aftersubroot = linenodes[currid + 1]
            setProb(treeroot, 2)
            addter(treeroot)
            if subroot is None:
                continue
            setProb(treeroot, 2)
            if subroot is not None:
                setProb(subroot, 1)
            if aftersubroot is not None:
                setProb(aftersubroot, 1)
            if presubroot is not None:
                setProb(presubroot, 1)
            cid = set(containID(subroot))
            minl, maxl = 1e10, -1
            for l in cid:
                maxl = max(maxl, l - 1)
                minl = min(minl, l - 1)
            precode = "\n".join(liness[0:minl])
            aftercode = "\n".join(liness[maxl + 1:])
            oldcode = "\n".join(liness[minl:maxl + 1])
            troot, vardic, typedic = solveLongTree(treeroot, subroot)
            data.append({'code':liness, 'treeroot':treeroot, 'troot':troot, 'oldcode':oldcode, 'filepath':filepath, 'subroot':subroot, 'vardic':vardic, 'typedic':typedic, 'idss':idss, 'classname':classname, 'precode':precode, 'aftercode':aftercode, 'tree':troot.printTreeWithVar(troot, vardic), 'prob':troot.getTreeProb(troot), 'mode':0, 'line':lineid, 'isa':False})
        solveone(data, model, bugid, 'fixed')