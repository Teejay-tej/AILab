##UCS


frontier=[]

def ucs(goal):
    global frontier
    global graph
    global flag
    while frontier:
        whole=sorted(frontier)[0]
        frontier.remove(whole)
        node=whole[1]
        cost=whole[0]
        if node ==  goal:
            print(cost)
            return
        for a in graph[node]:
            frontier.append((cost+graph[node][a],a))
            
            

graph={
    
    'a':{'b':3,'c':20,'d':222},
    'b':{'a':3,'d':20},
    'c':{'d':2},
    'd':{}
}
