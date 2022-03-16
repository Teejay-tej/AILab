##DLS

from collections import defaultdict
class Graph:
    def __init__(self,vertices):
        self.V=vertices
        self.graph=defaultdict(list)
    def addEdge(self,u,v):
        self.graph[u].append(v)
    def DLS(self,source,target,maxDepth):
        if source==target:
            return True
        if maxDepth<=0:
            return False
        for i in self.graph[source]:
            if (self.DLS(i,target,maxDepth-1)):
                return True
        return False

g=Graph(6)
g.addEdge(1,2)
g.addEdge(1,3)
g.addEdge(2,4)
g.addEdge(2,5)
g.addEdge(3,6)

target=6;
source=1;
maxDepth=2

if g.DLS(source,target,maxDepth)==True:
    print('target node is reachable from source node')
else:
    if g.DLS(source,target,maxDepth)==False:
        print('target NOT reachable from source node')
        
