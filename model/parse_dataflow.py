from copy import deepcopy
import numpy as np


terminalsFather = ["value", "name", "member", "qualifier"]
Statement = ["IfStatement", "then_statement", "cases", "else_statement", "AssertStatement", "SwitchStatement", "WhileStatement", "DoStatement", "ForStatement", "BreakStatement", "ContinueStatement", "ReturnStatement", "ThrowStatement", "SynchronizedStatement", "TryStatement", "StatementExpression"]

class Node:
    def __init__(self, name):
        self.name = name
        self.index = -1
        self.child = []
        self.flow = -1
        self.avaTerminals = []
        self.terminals = []

    def getavaterms(self):
        if len(self.child) > 0 and self.name in ["Assignment"]:
            return [self.terminals[0]]
        if len(self.child) > 0 and self.name in ["parameters"]:
            return self.terminals
        if len(self.child) > 0 and self.name in ["type"]:
            return []
        if len(self.child) > 0 and self.name in ["VariableDeclarator"]:
            return [self.terminals[0]]
        if len(self.child) > 0 and self.name in ["VariableDeclaration", "LocalVariableDeclaration"]:
            return [self.terminals[0]]
        for c in self.child:
            self.avaTerminals += c.getavaterms()
        return self.avaTerminals 

    def getTerminals(self, fatherName = ""):
        if len(self.child) == 0 and fatherName in terminalsFather:
            if fatherName == "qualifier" and self.name == "None":
                return []
            self.terminals += [[self.index, self.name]]
            return self.terminals
        for c in self.child:
            self.terminals += c.getTerminals(self.name)        
        if len(self.child) != 0 and self.name in ["MethodInvocation"]:
            return [self.terminals[-1]]
        if len(self.child) != 0 and self.name in ["type"]:
            return []

        return self.terminals

    def parseRootFlow(self, mat):
        for i in range(min(500, len(self.terminals))):
            for t in range(i - 1, -1, -1):
                if self.terminals[i][1] == self.terminals[t][1]:
                    a = self.terminals[i][0]
                    b = self.terminals[t][0]
                    mat[a][b] = mat[b][a] = 1

        return mat

    def parseFlow(self, mat):
        if len(self.child) == 0:
            return mat
        if len(self.child) != 0 and self.name in ["type"]:
            return []

        return self.terminals

    def parseRootFlow(self, mat):
        for i in range(min(500, len(self.terminals))):
            for t in range(i - 1, -1, -1):
                if self.terminals[i][1] == self.terminals[t][1]:
                    a = self.terminals[i][0]
                    b = self.terminals[t][0]
                    mat[a][b] = mat[b][a] = 1

        return mat

    def parseFlow(self, mat):
        if len(self.child) == 0:
            return mat
        if self.name == "Assignment": 
            now = self.terminals[0][0]
            for i in range(1, len(self.terminals)):
                flow = self.terminals[i][0]
                if now < 500 and flow < 500:
                    mat[now][flow] = mat[flow][now] = 1
        
        if self.name in ["MethodInvocation"]:
            now = self.terminals[-1][0]
            for i in range(len(self.terminals) - 1):
                flow = self.terminals[i][0]
                if now < 500 and flow < 500:
                    mat[now][flow] = mat[flow][now] = 1
        
        for c in self.child:
            mat = c.parseFlow(mat)

        return mat

    def print2Tree(self):
        s = str(self.name)
        for i in range(len(self.child)):
            s += " " + self.child[i].print2Tree()
        s += " ^"
        return s
    

flowData = ["Assignment"]
ParsePointer = 0

def parseTree2Node(tokens):
    global ParsePointer
    newnode = Node(tokens[ParsePointer])
    newnode.index = ParsePointer
    ParsePointer += 1
    while tokens[ParsePointer] != "^":
        newnode.child.append(parseTree2Node(tokens))
        ParsePointer += 1
    return newnode
    
def getTreeStruct(tokens):
    global ParsePointer 
    ParsePointer = 0
    root = parseTree2Node(tokens)
    return root

def parseTree(line):
    tokens = line.strip().split()
    index = []
    flist = []
    terminals = []
    nodefatherlist = []
    depth = 0
    for i in range(len(tokens)):
        if tokens[i] == "^":
            depth -= 1
            flist = flist[:-1]
            index.append(-1)
            terminals.append(0)
            nodefatherlist.append([])
        else:
            nodefatherlist.append(deepcopy(flist))
            if i == 0:
                index.append(-1)
            else:
                index.append(flist[-1])
            flist.append(i)
            if tokens[i + 1] == "^":
                if len(flist) >= 2 and tokens[flist[-2]] in terminalsFather:
                    terminals.append(1)
                else:
                    terminals.append(0)
            else:
                terminals.append(0)
            depth += 1

    root = getTreeStruct(tokens)
    print (root.print2Tree())

    return tokens, index, terminals, nodefatherlist, root

def findFather(list1, list2):
    for i in range(min(len(list1), len(list2))):
        if list1[i] != list2[i]:
            return list1[i - 1]

def GetFlow(line):
    tokens, index, terminals, nodefatherlist, root = parseTree(line)
    terminalsIndex = []
    root.getTerminals()
    root.getavaterms()
    terminals = [0 for i in range(len(terminals))]
    for i in root.terminals:
        terminals[i[0]] = 1
    avalist = [root.avaTerminals[i][0] for i in range(len(root.avaTerminals))]
    flowMat = np.zeros([500, 500])
    flowMat = root.parseFlow(flowMat)
    stmtlist = []
    bf = []
    af = []
    depth = 0
    for i in range(len(tokens)):
        token = tokens[i] 
        if terminals[i] == 1:
            for t in bf:
                if tokens[t] == tokens[i] and t < 500 and i < 500:
                    flowMat[t][i] = flowMat[i][t] = 1
        if i in avalist:
            newaf = []
            for t in af:
                if tokens[t] != tokens[i]:
                    newaf.append(t)
            newaf.append(i)
            af = newaf
            if len(stmtlist) == 0:
                bf = af
        if token == "^":
            depth -= 1
            if len(stmtlist) == 0:
                continue
            
            if depth < stmtlist[-1][0]:
                if stmtlist[-1][1] in ["IfStatement", "SwitchStatement"]:
                    newaf = []
                    for l in stmtlist[-1][3]:
                        for t in l:
                            if t not in newaf:
                                newaf.append(t)
                    af = newaf
                stmtlist = stmtlist[:-1]
                bf = af
                if len(stmtlist) == 0:
                    continue
                if stmtlist[-1][1] in ["IfStatement", "SwitchStatement"]:
                    stmtlist[-1][3].append(af)
                    bf = af = stmtlist[-1][2]
        else:
            depth += 1
        if token in Statement:
            stmtlist.append([depth, token, af, []])

    flow = []
    for i in range(500):
        for t in range(i + 1, 500):
            if flowMat[i][t] == 1:
                print (tokens[i], tokens[t])
                flow.append([i, t])

    return flow