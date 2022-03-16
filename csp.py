##CSP


def CSP(list1,tables):
  for i in list1:
    if i in tables:
      return False
  return True

n=int(input("enter number of people"))
c=int(input("Enter the no of tables"))
list1={}
tables={}
for i in range(c):
  tables[i]=[]
for i in range(n):
  list1[i]=[]

print("give the people who cant sit with each other: Type EXIT to exit")
while True:
  choice=input()
  if choice=="EXIT" or choice=="exit":
    break
  a,b=map(int,choice.split())
  list1[a].append(b)
  list1[b].append(a)

flag1= True
for i in range (n):
  flag=False
  for j in range(c):
    if CSP(list1[i],tables[j]):
      flag= True
      tables[j].append(i)
      break
  if flag==False:
    print("No arrangements can be made for the given constraints")
    flag1=False
    break
if flag1:
  print(tables)
