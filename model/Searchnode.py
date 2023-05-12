class Node:
    def __init__(self, name, d):
        self.name = name
        self.id = d
        self.father = None
        self.child = []
        self.sibiling = None
        self.expanded = False
        self.fatherlistID = 0
        self.treestr = ""
        self.block = ""
        self.num = 0
        self.fname = ""
        self.position = None
        self.possibility = 0

    def printTree(self, r):
      s = r.name + "" + " "
      if len(r.child) == 0:
        s += "^ "
        return s
      for c in r.child:
        s += self.printTree(c)
      s += "^ "
      
      return s

    def getNum(self):
        return len(self.getTreestr().strip().split())

    def getTreeProb(self, r):
      ans = [r.possibility]
      if len(r.child) == 0:
        return ans
      for c in r.child:
        ans += self.getTreeProb(c)
      
      return ans

    def getTreestr(self):
        if self.treestr == "":
            self.treestr = self.printTree(self)
            return self.treestr
        else:
            return self.treestr

    def printTreeWithVar(self, node, var):
        ans = ""
        if node.name in var:
            ans += var[node.name] + " "
        else:
            ans += node.name + " "
        for x in node.child:
            ans += self.printTreeWithVar(x, var)
        ans += '^ '  
        
        return ans

    def printTreeWithLine(self, node):
        ans = ""
        if node.position:
            ans += node.name + "-" + str(node.position.line)
        else:
            ans += node.name + "-"
        for x in node.child:
            ans += self.printTreeWithLine(x)
        ans += '^ '  
        
        return ans

    def printprob(self):
        ans = self.name + str(self.possibility) + ' '
        for x in self.child:
            ans += x.printprob()
        ans += '^ '
        
        return ans

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.name.lower() != other.name.lower():
            return False
        if len(self.child) != len(other.child):
            return False
        if True:
            return self.getTreestr().strip() == other.getTreestr().strip()
