class Operation:
    def __init__(self, index, op_name, op_type, op_group, op_next_id, op_output_shape=()):
        self.index = index
        self.name = op_name
        self.type = op_type
        self.group = op_group
        self.next = op_next_id
        self.output_shape = op_output_shape

class unionfind:
    def __init__(self, groups):
        self.groups=groups
        self.items=[]
        for g in groups:
            self.items+=list(g)
        self.items=set(self.items)
        self.parent={}
        self.rootdict={}
        for item in self.items:
            self.rootdict[item]=1
            self.parent[item]=item

    def union(self, r1, r2):
        rr1=self.findroot(r1)
        rr2=self.findroot(r2)
        cr1=self.rootdict[rr1]
        cr2=self.rootdict[rr2]
        if cr1>=cr2:
            self.parent[rr2]=rr1
            self.rootdict.pop(rr2)
            self.rootdict[rr1]=cr1+cr2
        else:
            self.parent[rr1]=rr2
            self.rootdict.pop(rr1)
            self.rootdict[rr2]=cr1+cr2

    def findroot(self, r):
        if r in self.rootdict.keys():
            return r
        else:
            return self.findroot(self.parent[r])

    def createtree(self):
        for g in self.groups:
            if len(g)< 2:
                continue
            else:
                for i in range(0, len(g)-1):
                    if self.findroot(g[i]) != self.findroot(g[i+1]):
                        self.union(g[i], g[i+1])

    def printree(self):
        rs={}
        for item in self.items:
            root=self.findroot(item)
            rs.setdefault(root,[])
            rs[root]+=[item]
        for key in rs.keys():
            print(rs[key])
        return rs