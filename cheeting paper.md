

# Cheeting Paper

### 施广熠 生命科学学院 数据结构与算法B

### 1、函数

```python
#二进制八进制与十六进制
ans=[2,12,34,99,-1,-7]
for k in range(len(ans)):
    print(bin(ans[k]),oct(ans[k]),hex(ans[k]))
#注意输出的数为“0b1100”，“0o14”，”0xc“

print(int("1010",2)) #int函数可将字符串类型的某进制数（参数即为数的进制）转化为十进制整型
```

```python
#需要赋值的函数：upper、lower、sorted、split、pop
#不需要赋值的函数：add、append、remove、sort

ord()  #返回ASC码值
chr()  #ASC码返回字符串
```

```python
print("string","list",sep=".",end="?") #控制间隔/结尾
#end=的默认值为"/n" 即换行 "/t" 为制表

#round()保留小数
print(round(3.123456789,5)# 3.12346
```

```python
#指定位数的小数保留
num=3.1415926
print(format(num, ".5f"), end="")
#如何在字符串中穿插变量
st1="hello world"
print(f"say:{st1}")
```

```python
#list.sort()的key参数
input()
lt = input().split()
max_len = len(max(lt, key = lambda x:len(x))) #x可替换为任意变量
lt.sort(key = lambda x: tuple([int(i) for i in x]) * ceil(max_len/len(x)))
lt1 = lt[::-1]
print(''.join(lt1),''.join(lt))
```

```py
#try/except
#邮箱验证
a=0
while True:
    try:
        e=input()
        if "@" in e and "@." not in e :
            a+=1
        if ".@" not in e and "." in e[e.find("@"):]:
            a+=1
        if e[0]!="@" and e[0]!=".":
            a+=1
        if e[-1]!="@" and e[-1]!=".":
            a+=1
        if e.count("@")==1:
            a+=1
        if a==5:
            a=0
            print("YES")
        else:
            a=0
            print("NO")
    except EOFError: #检测到EOFError表面测试数据结束
        break
```

### 2、数组

```python
#字符串
#字符串-大小写转化
letter="L" #判断字母大小写
print(letter.isupper(),letter.islower())
string="Good!"  #仅将小写字母转为大写/仅将大写字母转为小写
print(string.lower(),string.upper())
sent="it is SO GOOD!"  #转换为首字母大写/大小写转换
print(sent.capitalize(),sent.swapcase())

#题目【多项式的时间复杂度】
#字符串处理常用方法：截取[:]字符串，split("?")分割，+直接相加
strings=input().split('+')
tuple_strings=[(s.split('n^')) for s in strings]#同时存储系数和次数
maximum=0
for string in tuple_strings:
    if string[0]!='0':
        maximum=max(maximum,int(string[1]))#迭代器遍历，程序更舒服
print('n^'+str(maximum))
```

```python
#字典相关
d={"k1":1} #新建一个字典
d["k2"]=2  #修改字典“k2”键对应的值，如无对应的键，则创建新的键值对
k1=d["k1"] #访问k1键对应的值
k2=d.get("k2","NO")
k3=d.get("k3","NO")  #get()访问时，如果没有找到键，则返回默认值用","加在后面
l1=d.values()  #返回所有值的列表（直接print会有问题），同理还有keys/items
print(2 in l1)
#遍历方法：
for key,value in d.items():
    print(key,value)
del d["k1"]
a=d.pop("k2")  #弹出给定键的相应值，在字典中删除键值对
d={"k1":4,"k2":5,"k3":6}
b=d.popitem()  #弹出最后一个键值对（以元组形式）
```

```python
#集合相关
#空集合用s=set()
SET={1,3,2,4,5,4}  #区别于元组，集合既不能切片，也不能直接相加
print(SET)   #
SET.add(4)   #集合不能重复，自动归到同一个数字
SET.add(6)
SET.remove(2) #或用discard()，不会报错
SET.pop()  #从最右端弹出
print(SET)
set0={0,1,2,3}
print(set0 & SET,set0 | SET,set0 - SET)
#分别为交集/并集/补集

#题目【校门口的树】
L,n=map(int,input().split())
s=set()     #空集合为s=set();s={}为创建一个空的字典
for i in range(n):
    start,end=map(int,input().split())
    se={i for i in range(start,end+1)}  #集合生成式的使用
    s=s.union(se)  #union函数表示取并集，返回一个集合
print(L-len(s)+1)
```

```py
#双向链表 （增删复杂度为O（1））
from collections import deque
l=[2,3,4,5]
q=deque(l)  #创建双向列表
#appendleft()/append() 向队头/队尾添加元素,添加序列则为extendlrft()/extend()
q.insert(2,3.5) #向指定位置插入元素（前一个参数为index）
#popleft()/pop() 从队头/队尾弹出元素
#index() count()均可使用
```

```py
#堆排序
import heapq
heap=[3,22,5,5,1,3]
heapq.heapify(heap)  #原地转化，不需要赋值
heapq.heappush(32)  #增添元素，保持堆的合法性
heapq.heappop()  #pop堆中最小的元素，创建heap时所有元素取反，弹出之后再取反就可以pop最大值
heapq.nlargest(2,heap)  #弹出最大的n个，最小n个用nsmallest()
#heapreplace(heap,item)先弹出最小元素，再添加item
#heappushpop(heap,item)先添加item，再弹出最小元素

#12.剪绳子
import heapq  #用堆的结构，效率非常高
n=int(input())
num=[int(i) for i in input().split()]
heapq.heapify(num)
su=0
while len(num)>1:
    num1=heapq.heappop(num)
    num2=heapq.heappop(num)
    su+=num1+num2
    heapq.heappush(num,num1+num2)
print(su)
```

```py
#栈 题目【波兰表达式】
# 用栈解决，更好理解
expression = input().split()
stack = []
ind=len(expression)-1
for i in range(1,len(expression)+1):
    a = expression[-i]
    if a in ['+', '-', '*', '/']:
        c = stack.pop(-1)
        d = stack.pop(-1)
        if a == '+':
            stack.append(c + d)
        elif a == '-':
            stack.append(c - d)
        elif a == '*':
            stack.append(c * d)
        else:
            stack.append(c / d)
    else:
        stack.append(float(a))

print("{:.6f}".format(stack[0]))
```



### 3、工具

ASCⅡ码表

![image-20231227223041869](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20231227223041869.png)

```python
#埃氏筛
def prime(n):
    s=set()
    ans=0
    ans=[True]*(n+1)
    ans[0]=ans[1]=False
    for i in range(2,int(n**0.5)+1):
        if ans[i]:
            for j in range(i*i,n+1,i):
                ans[j]=False
    for k in range(2,n+1):
        if ans[k]:
            s.add(k)
    return s
```

```py
#二分查找与插入
import bisect
sorted_list = [1,3,5,7,9] #[(0)1, (1)3, (2)5, (3)7, (4)9]
position = bisect.bisect_left(sorted_list, 6)
print(position) # 输出：3，因为6应该插入到位置3，才能保持列表的升序顺序

bisect.insort_left(sorted_list, 6)
print(sorted_list) # 输出：[1, 3, 5, 6, 7, 9]，6被插入到适当的位置以保持升序顺序

sorted_list=(1,3,5,7,7,7,9) #left为小于等于x的第一个索引，right为大于x的第一个索引
print(bisect.bisect_left(sorted_list,7))
print(bisect.bisect_right(sorted_list,7))
# 输出：3 6

#二分查找源码
def bisect_right(a, x, lo=0, hi=None, *, key=None):

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    if key is None:
        while lo < hi:
            mid = (lo + hi) // 2
            if x < a[mid]:
                hi = mid
            else:
                lo = mid + 1
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if x < key(a[mid]):
                hi = mid
            else:
                lo = mid + 1
    return lo
```

```python
#calendar（improt calendar)
1. calendar.month(年, 月) : 返回一个月份的日历字符串。它接受年份和月份作为参数，并以多行
字符串的形式返回该月份的日历。
2. calendar.calendar(年) : 返回一个年份的日历字符串。这个函数生成整个年份的日历，格式化为
多行字符串。
3. calendar.monthrange(年, 月) : 返回两个整数，第一个是该月第一天是周几（0-6表示周一到周
日），第二个是该月的天数。
4. calendar.weekday(年, 月, 日) : 返回给定日期是星期几。0-6的返回值分别代表星期一到星期
日。
5. calendar.isleap(年) : 返回一个布尔值，指示指定的年份是否是闰年。
6. calendar.leapdays(年1, 年2) : 返回在指定范围内的闰年数量，不包括第二个年份。
7. calendar.monthcalendar(年, 月) : 返回一个整数矩阵，表示指定月份的日历。每个子列表表示
一个星期；天数为0表示该月份此天不在该星期内。
8. calendar.setfirstweekday(星期) : 设置日历每周的起始日。默认情况下，第一天是星期一，但
可以通过这个函数更改。
9. calendar.firstweekday() : 返回当前设置的每周起始日。
"""
```

```python
#Counter
from collections import Counter
# O(n)
# 创建一个待统计的列表
data = ['apple', 'banana', 'apple', 'orange', 'banana', 'apple']
# 使用Counter统计元素出现次数
counter_result = Counter(data) # 返回一个字典类型的东西
# 输出统计结果
print(counter_result) # Counter({'apple': 3, 'banana': 2, 'orange': 1})
print(counter_result["apple"]) # 3
```

```py
#排列与组合
from itertools import permutations as per
elements = [1, 2, 3]
permutations = list(per(elements))
#[(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]

from itertools import combinations as com
elements = ['A', 'B', 'C', 'D']
# 生成所有长度为2的组合
combinations = list(com(elements, 2))
#[('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'D')]
```



### 4、模板与例题

#### dp:

```py
#采药-01背包-一维形式
#背包问题可以从二维本质出发
T,M=map(int,input().split())
l=[]
dp=[0]*(T+1)
for i in range(M):
    t,v=map(int,input().split())
    l.append((t,v))
for k in range(M):
    for j in range(T,0,-1):
        if j>=l[k][0]:
            dp[j]=max(dp[j],l[k][1]+dp[j-l[k][0]])
print(dp[-1])

#18.最佳凑单 #01背包问题，理解dp[i,j]的意义
n,t=map(int,input().split())
value_list=[int(_) for _ in input().split()]
dp=[999999999]*(t+1)
for i in range(n):
    for j in range(t,-1,-1):
        if value_list[i]>=j:
            dp[j]=min(dp[j],value_list[i]-j)
        else:
            dp[j]=min(dp[j],dp[j-value_list[i]])
if dp[-1]==999999999:
    print(0)
else:
    print(dp[-1]+t)
```

```py
#完全背包(初始版本/一维数组优化/状态转移方程优化）
#多重背包就对每个i以最后一个个数参数增加一次循环
n,v=map(int,input().split())
staff=[]
for i in range(n):
    V,W=map(int, input().split())
    staff.append((V,W))
dp=[[0]*(v+1) for i in range(n+1)]
for i in range(1,n+1):
    for j in range(1,v+1):
        if j<staff[i-1][0]:
            dp[i][j]=dp[i-1][j]
        else:
            k=1
            while j>=k*staff[i-1][0]:
                dp[i][j]=max(dp[i-1][j-k*staff[i-1][0]]+k*staff[i-1][1],dp[i-1][j],dp[i][j])
                k+=1
print(dp[-1][-1])

n,v=map(int,input().split())
staff=[]
for i in range(n):
    V,W=map(int, input().split())
    staff.append((V,W))
dp=[0]*(v+1)
for i in range(1,n+1):
    for j in range(v,0,-1):
        k=0
        while j>=k*staff[i-1][0]:
            dp[j]=max(dp[j-k*staff[i-1][0]]+k*staff[i-1][1],dp[j])
            k+=1
print(dp[-1])

n,v=map(int,input().split())
staff=[]
for i in range(n):
    V,W=map(int, input().split())
    staff.append((V,W))
dp=[[0]*(v+1) for i in range(n+1)]
for i in range(1,n+1):
    for j in range(1,v+1):
        if j>=staff[i-1][0]:  #相当于取了某些个数的i，先返回同行位置，取不了了再回到i-1
            dp[i][j]=max(dp[i][j-staff[i-1][0]]+staff[i-1][1],dp[i-1][j])
        else:
            dp[i][j]=dp[i-1][j-1]
print(dp[-1][-1])

#多重背包还可以视作增加了一样物品的0-1背包，即

# n, v分别代表物品数量，背包容积
n, v = map(int, input().split())
# w为物品价值，c为物品体积（花费）
w, cost, s = [0], [0], [0]
for i in range(n):
cur_c, cur_w,cur_s= map(int, input().split())
w += [cur_w]*cur_s
cost += [cur_c]*cur_s
```

```py
#10.剪彩
n,a,b,c=map(int,input().split())
dp=[0]+[-4000]*n  #保证只能从0发出
for i in range(n+1):
    for j in (a,b,c):
        if i>=j:
            dp[i]=max(dp[i],dp[i-j]+1)
print(dp[n])
```

```py
#3.合唱队形  寻找最长递增/递减子列
#一维dp，dp[i]表示包含i的从一端开始的最长递增子列，便于得出状态转移方程
n=int(input())
height=[int(i) for i in input().split()]
dp_l=[1]*n
for i in range(1,n):
    for j in range(i):
        if height[i]>height[j]:
            dp_l[i]=max(dp_l[i],dp_l[j]+1)
dp_r=[1]*n
for i in range(n-2,-1,-1):
    for j in range(n-1,i,-1):
        if height[i]>height[j]:
            dp_r[i]=max(dp_r[i],dp_r[j]+1)
ma=0
for i in range(n):
    ma=max(ma,dp_l[i]+dp_r[i])
print(n-ma+1)
```



#### 图

##### bfs:

```py
#寻宝 bfs版，bfs寻找最短路径长度
from collections import deque
def valid(x,y):
    if 0<=x<m and 0<=y<n:
        if treasure[x][y]!=2 and not inque[x][y]:
            return True
        else:
            return False
    else:
        return False

def bfs():
    q=deque()
    q.append((0,0,0)) #将目前格子的步数和位置一同记录
    while q:
        t=q.popleft()
        for i in range(4):
            x,y,cnt=t[0]+dx[i],t[1]+dy[i],t[2]
            if valid(x,y):
                inque[x][y]=True
                q.append((x,y,cnt+1)) #在前一格的基础上cnt+1
                if treasure[x][y]==1:
                    print(cnt+1)
                    return
    print("NO")

m,n=map(int,input().split())
treasure=[]
for i in range(m):
    x=[int(_) for _ in input().split()]
    treasure.append(x)
inque=[[False]*n for i in range(m)]
dx=[0,1,0,-1]
dy=[1,0,-1,0]
if treasure[0][0]==1:
    print(0)
    exit()
if treasure[0][0]==2:
    print("NO")
    exit()
bfs()
```

```py
#最大连通域面积-bfs模板-面积计算
from collections import deque
dx=[1,-1,0,0,1,1,-1,-1]
dy=[0,0,1,-1,1,-1,-1,1]
max_total=0
def valid(x,y):
    if 0<=x<n and 0<=y<m:
        if matrix[x][y]=="W" and not inq[x][y]:
            return True
        else:
            return False
    else:
        return False
def bfs(x,y):
    global max_total  #多组数据时全局变量一定不要忘记归零！
    q=deque()
    q.append((x,y))
    inq[x][y]=True
    cnt=1
    while q:
        bounce=q.popleft()
        for r in range(8):
            nx=bounce[0]+dx[r]
            ny=bounce[1]+dy[r]
            if valid(nx,ny):
                q.append((nx,ny))
                inq[nx][ny]=True
                cnt+=1
    max_total=max(max_total,cnt)

t=int(input())
for i in range(t):
    n,m=map(int,input().split())
    matrix=[]
    for i0 in range(n):
        matrix.append(input())
    inq=[]
    for i1 in range(n):
        inq.append([False]*m)
    for k1 in range(n):
        for k2 in range(m):
            if valid(k1,k2):
                bfs(k1,k2)
    print(max_total)
    max_total=0
```

```py
#迷宫最短路径
#如何在bfs中记录路径
from queue import Queue
MAXN = 100
MAXD = 4
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]
def canVisit(x, y):
    return x >= 0 and x < n and y >= 0 and y < m and maze[x][y] == 0 and not inQueue[x][y]
def BFS(x, y):
    q = Queue()
    q.put((x, y))
    inQueue[x][y] = True
    while not q.empty():
        front = q.get()
        if front[0] == n - 1 and front[1] == m - 1:
            return
        for i in range(MAXD):
            nextX = front[0] + dx[i]
            nextY = front[1] + dy[i]
            if canVisit(nextX, nextY):
                pre[nextX][nextY] = (front[0], front[1]) #pre中每个点都记录前一个点的位置
                inQueue[nextX][nextY] = True
                q.put((nextX, nextY))
def printPath(p):
    prePosition = pre[p[0]][p[1]]
    if prePosition == (-1, -1):
        print(p[0] + 1, p[1] + 1)
        return
    printPath(prePosition)  #类似回溯，推到起始点时开始从头输出
    print(p[0] + 1, p[1] + 1)

n, m = map(int, input().split())
maze = []
for _ in range(n):
    row = list(map(int, input().split()))
    maze.append(row)
inQueue = [[False] * m for _ in range(n)]
pre = [[(-1, -1)] * m for _ in range(n)]
BFS(0, 0)
printPath((n - 1, m - 1))
```

##### Dijkstra

```python
#通过堆排序的Dijkstra算法（某点到指定顶点的最小路径）
#兔子与樱花
import heapq
def dijkstra(graph,start,end):
    if start == end: 
        return []
    dist = {i:(99999999,[]) for i in graph} #dist字典用于储存顶点名，与起点到顶点的最短距离（权值）以及[]中的路径
    dist[start] = (0,[start])
    pos = [] 
    heapq.heappush(pos,(0,start,[]))
    while pos:
        dist1,current,path = heapq.heappop(pos) #dist1表示起点到上一个相邻节点的距离
        for (next,dist2) in graph[current].items():
            if dist2+dist1 < dist[next][0]:
                dist[next] = (dist2+dist1,path+[next])
                heapq.heappush(pos,(dist1+dist2,next,path+[next]))
    return dist[end][1]

P = int(input())
graph = {input():{} for _ in range(P)} #构建出的图形式为双重字典，每个key代表图中的node，value为键节点所对应所有节点的字典，字典中为对应的路径与权值。
for _ in range(int(input())):
    place1,place2,dist = input().split()
    graph[place1][place2] = graph[place2][place1] = int(dist)

for _ in range(int(input())):
    start,end = input().split()
    path = dijkstra(graph,start,end)
    s = start
    current = start
    for i in path:
        s += f'->({graph[current][i]})->{i}'
        current = i
    print(s)
```

##### dfs:

```py
#寻宝 dfs版
def dfs(x,y):
    dx=[1,0,-1,0]
    dy=[0,1,0,-1]
    global cnt
    global cnt_min
    global treasure
    for i in range(4):
        if treasure[x+dx[i]][y+dy[i]]!=2:
            treasure[x][y]=2
            cnt+=1
            if treasure[x+dx[i]][y+dy[i]]==1:
                cnt_min=min(cnt,cnt_min)
                treasure[x][y]=0  #注意计数器和图都要恢复回溯
                cnt-=1
                break
            else:
                dfs(x+dx[i],y+dy[i])
                treasure[x][y]=0
                cnt-=1
    return

cnt_min=99999
cnt=0
n,m=map(int,input().split())
treasure=[[2]*(m+2)]
for i in range(n):
    treasure.append([2]+[int(j) for j in input().split()]+[2])
treasure.append([2]*(m+2))
if treasure[1][1]==1:
    print(0)
    exit()
dfs(1,1)
if cnt_min==99999:
    print("NO")
else:
    print(cnt_min)
```

```py
#八皇后问题
def dfs(cur):
    global solution
    for i0 in range(8):
        flag=True
        for j0 in range(len(solution)): #用列表solution直接储存某一次的结果，因为八皇后可以直接根据前面的路径判断后面的位置是否可访问，不需要在图上修改
            if cur+i0==j0+solution[j0] or cur-i0==j0-solution[j0] or i0 in solution:
                flag=False
        if flag:
            solution+=[i0]
            if cur==7:
                su=0
                for i1 in range(8):
                    su+=(solution[i1]+1)*10**(7-i1)
                ans.append(su)
                solution.pop(-1)
            else:
                dfs(cur+1)
                solution.pop(-1)
        else:
            continue

ans=[]
solution=[]
dfs(0)
n=int(input())
for i in range(n):
    num=int(input())
    print(ans[num-1])
```

```py
#组合乘积
def dfs(T,l): #dfs解决非图问题
    for i in range(len(l)):
        if l[i]!=0:
            if T%l[i]==0 and l[i]>0:
                T=T//l[i]
                temp=l[i]
                l[i]=-1  #防止路径重复
                if T==1:
                    print("YES")
                    exit()
                else:
                    dfs(T,l)
                    l[i]=temp  #零时储存前一个值便于回溯
                    T=T*temp

t=int(input())
num=[int(i) for i in input().split()]
if t==0:
    if 0 in num:
        print("YES")
    else:
        print("NO")
else:
    dfs(t,num)
    print("NO")
```

```python
#迷宫最大权值和
dx = [-1, 0, 1, 0]
dy = [ 0, 1, 0, -1]
maxValue = -9999
def dfs(maze, x, y, nowValue):
    global maxValue
    if x==n and y==m:
        if nowValue > maxValue:
            maxValue = nowValue
            return
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if maze[nx][ny] == 0:
            maze[nx][ny] = -1
            tmp = w[x][y]
            w[x][y] = -9999
            nextValue = nowValue + w[nx][ny]
            dfs(maze, nx, ny, nextValue)
            maze[nx][ny] = 0
            w[x][y] = tmp

n, m = map(int, input().split())
maze = []
maze.append( [-1 for x in range(m+2)] )
for _ in range(n):
    maze.append([-1] + [int(_) for _ in input().split()] + [-1])
maze.append([-1 for x in range(m + 2)])
w = []
w.append([-9999 for x in range(m + 2)])
for _ in range(n):
    w.append([-9999] + [int(_) for _ in input().split()] + [-9999])
w.append([-9999 for x in range(m + 2)])
dfs(maze, 1, 1, w[1][1])
print(maxValue)
```

##### 最小生成树：

```python
#兔子与星空

#prim算法
import heapq

def prim(graph, start):
    mst = []
    used = set([start]) #记录已经加入的edge
    edges = [
        (cost, start, to)
        for to, cost in graph[start].items()
    ]
    heapq.heapify(edges)  #对目前生成的树的邻接边进行排序

    while edges:
        cost, frm, to = heapq.heappop(edges)
        if to not in used:
            used.add(to)
            mst.append((frm, to, cost)) 
            for to_next, cost2 in graph[to].items(): #将与新加入的节点相连的边加入edges
                if to_next not in used:
                    heapq.heappush(edges, (cost2, to, to_next))

    return mst  # 得到两端点和相连边权重表示的 mst

def solve():
    n = int(input())
    graph = {chr(i+65): {} for i in range(n)} #双重字典外层
    for i in range(n-1):
        data = input().split()
        star = data[0] # "A"
        m = int(data[1]) # 邻接边数量
        for j in range(m):
            next_star = data[2+j*2]
            cost = int(data[3+j*2])
            graph[star][next_star] = graph[next_star][star] = cost #双重字典内层

    mst = prim(graph, 'A')
    print(sum(x[2] for x in mst)) #树中所有边权重和

solve()


#kruskal算法
class DisjSet:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0]*n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        xset, yset = self.find(x), self.find(y)
        if self.rank[xset] > self.rank[yset]:
            self.parent[yset] = xset
        else:
            self.parent[xset] = yset
            if self.rank[xset] == self.rank[yset]:
                self.rank[yset] += 1

def kruskal(n, edges):
    dset = DisjSet(n)
    edges.sort(key = lambda x:x[2]) #仅根据边的绝对权值进行排序
    sol = 0
    for u, v, w in edges:
        u, v = ord(u)-65, ord(v)-65 #为了使用并查集中的find函数，"A"->0,"B"->1
        if dset.find(u) != dset.find(v):
            dset.union(u, v)
            sol += w
    if len(set(dset.find(i) for i in range(n))) > 1:
        return -1
    return sol

n = int(input())
edges = []  #与prim不同，不在edges进行堆排序，而是直接将所有边加入edges
for _ in range(n-1):
    arr = input().split()
    root, m = arr[0], int(arr[1])
    for i in range(m):
        edges.append((root, arr[2+2*i], int(arr[3+2*i])))
print(kruskal(n, edges))

```

##### 拓扑排序

```python
from collections import deque, defaultdict

def topo_sort(vertices, edges):
    # 计算所有顶点的入度
    in_degree = {v: 0 for v in vertices}
    graph = defaultdict(list)

    # u->v
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1  # v的入度+1

    # 将所有入度为0的顶点加入队列
    queue = deque([v for v in vertices if in_degree[v] == 0])
    sorted_order = []

    while queue:
        u = queue.popleft()
        sorted_order.append(u)

        # 对于每一个相邻顶点，减少其入度
        for v in graph[u]:
            in_degree[v] -= 1
            # 如果入度减为0，则加入队列
            if in_degree[v] == 0:
                queue.append(v)

    if len(sorted_order) != len(vertices):
        return None  # 存在环，无法进行拓扑排序
    return sorted_order
#示例使用
vertices = ['A', 'B', 'C', 'D', 'E', 'F']
edges = [('A', 'D'), ('F', 'B'), ('B', 'D'), ('F', 'A'), ('D', 'C')]
result = topo_sort(vertices, edges)
if result:
    print("拓扑排序结果:", result)
else:
    print("图中有环，无法进行拓扑排序")

#输出所有的拓扑排序
from collections import deque
def all_topological_sorts(graph):
    # 计算所有节点的入度
    in_degree = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    # 找到所有入度为0的节点
    start_nodes = deque([k for k in in_degree if in_degree[k] == 0])
    result = []
    sort_helper(graph, in_degree, start_nodes, [], result)
    return result

def sort_helper(graph, in_degree, start_nodes, path, result):
    if not start_nodes:
        if len(path) == len(in_degree):
            result.append(list(path))
        return

    for node in list(start_nodes):
        # 将当前节点添加到路径中
        path.append(node)
        start_nodes.remove(node)
        # 减少当前节点指向的所有节点的入度
        for m in graph[node]:
            in_degree[m] -= 1
            if in_degree[m] == 0:
                start_nodes.append(m)
        # 递归调用
        sort_helper(graph, in_degree, start_nodes, path, result)
        # 回溯
        path.pop()
        start_nodes.append(node)
        for m in graph[node]:
            in_degree[m] += 1
            if in_degree[m] == 1:
                start_nodes.remove(m)

# 示例图
graph = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['D'],
    'D': []
}

# 打印所有拓扑排序
sorts = all_topological_sorts(graph)
for sort in sorts:
    if len(sort) != len(graph.keys()):
        print("not DAG")
        exit()
    print(sort)
```



#### 双指针：

```python
#三数之和，排序后用双指针解决
def threeSum(nums):
    nums.sort()  # 先对数组排序
    result = []
    n = len(nums)

    for i in range(n - 2):
        # 跳过重复的元素
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        # 双指针
        left = i + 1
        right = n - 1

        while left < right:
            total = nums[i] + nums[left] + nums[right]

            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])

                # 跳过重复的元素
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1

                left += 1
                right -= 1

    return len(result)

nums = [int(_) for _ in input().split()]
count = threeSum(nums)
print(count)

#朵拉找子列 子列末端都不是整个子段的最大值或者最小值
N=int(input())
for i in range(N):
    n=int(input())
    l=[int(j) for j in input().split()]
    if n<4:
        print(-1)
    else:
        left=0
        right=n-1  #因题目所给排列的特殊性，可以确认最大值和最小值，用双指针向中间取
        		   #尤其是最大的子列，是从全体中筛选出来的一段
        M=n
        m=1
        while right-left>=3:
            if l[left]==M:
                M-=1
                left+=1
            elif l[right]==M:
                M-=1
                right-=1
            elif l[left]==m:
                m+=1
                left+=1
            elif l[right]==m:
                m+=1
                right-=1
            else:
                print(left+1,right+1)
                exit()
        print(-1)
```



#### 逆向思维：

```py
#垃圾炸弹 mixtra
#反向思维，注意到垃圾的点数很少，可以将垃圾的值加到周围d*d的点上
d=int(input())
n=int(input())
square = [[0]*1025 for _ in range(1025)]
for _ in range(n):
    x,y,k=map(int, input().split())
    for i in range(max(x-d, 0), min(x+d+1, 1025)):
        for j in range(max(y-d, 0), min(y+d+1, 1025)):
            square[i][j] += k
res = max_point = 0
for i in range(0, 1025):
    for j in range(0, 1025):
        if square[i][j] > max_point:
            max_point = square[i][j]
            res=1
        elif square[i][j] == max_point:
            res+=1
print(res, max_point)
```

```py
#河中跳房子 用确定的interval长度反向确定移走的石头数
#注意二分查找的写法
n,m = map(int, input().split())
expenditure = []
for _ in range(n):
    expenditure.append(int(input()))

def check(x):
    num, s = 1, 0
    for i in range(n):
        if s + expenditure[i] > x:
            s = expenditure[i]
            num += 1
        else:
            s += expenditure[i]
    
    return [False, True][num > m]

lo = max(expenditure)
# hi = sum(expenditure)
hi = sum(expenditure) + 1
ans = 1
while lo < hi:
    mid = (lo + hi) // 2
    if check(mid):      # 返回True，是因为num>m，是确定不合适
        lo = mid + 1    # 所以lo可以置为 mid + 1。
    else:
        ans = mid    # 如果num==m, mid就是答案
        hi = mid
        
#print(lo)
print(ans)
```

#### 连续修改：

```py
#猫咪派对
#使用一个数组 f 来记录每种颜色出现的次数，使用另一个数组 cnt 来统计每个次数的颜色数量。
#通过迭代颜色列表，并根据不同的条件判断，计算并更新最长的连续天数 ans。
n = int(input())
colors = list(map(int, input().split()))

N = 10**5 + 10
ans = 0
mx = 0
f = [0] * N
cnt = [0] * N

for i in range(1, n + 1):
    color = colors[i - 1]
    cnt[f[color]] -= 1
    f[color] += 1
    cnt[f[color]] += 1
    mx = max(mx, f[color])
    ok = False
    if cnt[1] == i:  # every color has occurrence of 1
        ok = True
    elif cnt[i] == 1:  # only one color has the maximum occurrence and the occurrence is i
        ok = True
    elif cnt[1] == 1 and cnt[mx] * mx == i - 1:  # one color has occurrence of 1 and other colors have the same occurrence
        ok = True
    elif cnt[mx - 1] * (mx - 1) == i - mx and cnt[mx] == 1:  # one color has the occurrence 1 more than any other color
        ok = True
    if ok:
        ans = i

print(ans)
```



#### 余数判断整体：

```py
#XXXXX
line=int(input())
for i in range(line):
    n,x=map(int,input().split())
    l=input().split()
    l1=[]
    a=0
    for j in range(len(l)):
        l1.append(int(l[j]))
        if int(l[j])%x==0:
            a+=1
    if a==len(l1):
        print(-1)
    else:
        if sum(l1)%x!=0:
            print(n)
        else:
            for k in range(n):
               if l1[k]%x!=0 or l1[-(k+1)]%x!=0:
                   print(n-k-1)
                   break
```



#### 线性最优：

```py
#世界杯只因
n=int(input())
zy=[int(i) for i in input().split()]
rang=[(max(0,i-zy[i]),min(n-1,i+zy[i])) for i in range(n)]  #对监控的覆盖范围进行排序
rang.sort()
end=-1
r=0
num=0
while end<n-1:
    ma=0
    while rang[r][0]<=end+1:
        ma=max(ma,rang[r][1])  #找出覆盖不断情况下的最远覆盖末端（第一次要求从0开始，之后仅要求从上一次新增大的范围中选择最远覆盖末端）
        r+=1
        if r==n:
            break
    r=max(end,0) #从新一次的起点开始遍历
    end=ma  
    num+=1
print(num)
```

```py
#雷达安装
cnt=0
while True:
    cnt+=1
    n,d=map(int,input().split())
    if n==0:
        exit()
    island=[]
    interval=[]
    for i in range(n):
        a,b=map(int,input().split())
        island.append((a,b))
    flag=True
    #判断是否能覆盖
    for j in range(n):
        if island[j][1]>d:
            flag=False
    if not flag:
        print(f"Case {cnt}: -1")
        input()
        continue
    #####
    for i in range(n):  #将雷达位置转化为线性坐标上的interval区间
        s=island[i][0]-(d*d-island[i][1]**2)**0.5
        e=island[i][0]+(d*d-island[i][1]**2)**0.5
        interval.append((s,e))
    interval.sort()  #先进行排序，排序可以保证局部的处理方法是最优解
    su=1
    pos=[interval[0][0],interval[0][1]]
    for j in range(n):
        if interval[j][0]<=pos[1]:
            pos[1]=min(interval[j][1],pos[1])
        else:
            pos=[interval[j][0],interval[j][1]]
            su+=1
    print(f"Case {cnt}: {su}")
    input()
```

#### 子列：

```py
#15.分发糖果
n=int(input())  #本质上就是寻找单调递增或递减的子序列
l1=[int(i) for i in input().split()]
l2=[1]*n
l3=[1]*n
for i in range(1,n):
    if l1[i]>l1[i-1]:
        l2[i]=l2[i-1]+1
    else:
        l2[i]=1
for i in range(n-2,-1,-1):
    if l1[i]>l1[i+1]:
        l3[i]=l3[i+1]+1
    else:
        l3[i]=1
su=0
for i in range(n):
    su+=max(l2[i],l3[i])
print(su)
```



#### 链表成环

```python
#约瑟夫问题
from collections import deque
n,k=map(int,input().split())
queue=deque(i for i in range(1,n+1))
flag=k
res=[]
# 1 2 3 4 5 6 7 8 9 10
while len(queue)>=2:
    a=queue.popleft()
    queue.append(a)
    if k-2!=0:
        for _ in range(k-2):
            a = queue.popleft()
            queue.append(a)
    b=queue.popleft()
    res.append(b)
res_new=[str(i) for i in res]
print(" ".join(res_new))
```



#### 快速排序

```python
def quicksort(arr, left, right):
    if left < right:
        partition_pos = partition(arr, left, right)
        quicksort(arr, left, partition_pos - 1)
        quicksort(arr, partition_pos + 1, right)


def partition(arr, left, right):
    # 最右端元素作为基准元素，i,j是两个指针，通过两个元素的交换实现
    # 基准元素左右分别小于、大于他本身。
    i = left
    j = right - 1
    pivot = arr[right]
    while i <= j:
        while i <= right and arr[i] < pivot:
            i += 1
        while j >= left and arr[j] >= pivot:
            j -= 1
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]
    if arr[i] > pivot:
        arr[i], arr[right] = arr[right], arr[i]
    return i


arr = [22, 11, 88, 66, 55, 77, 33, 44]
quicksort(arr, 0, len(arr) - 1)
print(arr)

# [11, 22, 33, 44, 55, 66, 77, 88]
```



#### 并查集：

```python
#发现他，抓住他（识别案件是否为同一犯罪团伙所为）
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n)) # 初始化时，每个元素的父节点为自己
        self.rank = [0] * n # 初始化时每棵树的深度为1

    def find(self, x): #若当前树的parent不是自己，则向上寻找根节点
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]: #较浅的根节点连到较深的根节点
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

def solve():
    n, m = map(int, input().split())
    uf = UnionFind(2 * n)  # 初始化并查集，每个案件对应两个节点
    for _ in range(m):
        print(uf.parent,uf.rank)
        operation, a, b = input().split()
        a, b = int(a)-1, int(b)-1
        if operation == "D":
            uf.union(a, b+n)  # a与b的对立案件合并
            uf.union(a+n, b)  # a的对立案件与b合并
        else:  # "A"
            if uf.find(a) == uf.find(b) or uf.find(a + n) == uf.find(b + n):
                print("In the same gang.")
            elif uf.find(a) == uf.find(b + n) or uf.find(a + n) == uf.find(b):
                print("In different gangs.")
            else:
                print("Not sure yet.")

T = int(input())
for _ in range(T):
    solve()
    
#兔子与星空：kruskal算法中区分是否为联通量
```

#### 树：

```python
#二叉树的高度和叶子数量
class TreeNode:
    def __init__(self):
        self.left=None
        self.right=None #定义树的节点这一数据类型

def height(node):  #二叉树高度
    if node==None:  #在普通函数中用数据类型名代指这一类型的数据
        return -1
    else:
        return max(height(node.left)+1,height(node.right)+1)

def leaves(node):  #叶子数量
    if node==None:
        return 0
    elif node.left==None and node.right==None:
        return 1
    else:
        return leaves(node.left)+leaves(node.right)

n=int(input())
nodes=[TreeNode() for i in range(n)]  #将整棵树储存在“节点“列表中，注意node()的写法
for i in range(n):
    l,r=map(int,input().split())
    if l!=-1:
        nodes[i].left=nodes[l-1]  #将列表的节点按照题目数据连接起来
    if r!=-1:
        nodes[i].right=nodes[r-1]
print(height(nodes[0]))
print(leaves(nodes[0]))

#若节点不是按照顺序排号，仅给出相互连接的关系，则需要找出根节点(root)

has_parent = [False] * n  # 用来标记节点是否有父节点

for i in range(n):
    left_index, right_index = map(int, input().split())
    if left_index != -1:
        has_parent[left_index] = True
    if right_index != -1:
        has_parent[right_index] = True

# 寻找根节点，也就是没有父节点的节点
root_index = has_parent.index(False)
root = nodes[root_index]

#用dfs方法遍历树（图的四个方向变为二叉树的两个方向）
#二叉树的深度
def dfs(num):
    global deep,deepest #声明全局变量
    path=[0,1]
    for i in range(2):
        if nodes[num][i] != -1:
            deep+=1
            deepest=max(deep,deepest)
            dfs(nodes[num][i]-1)
            deep-=1
    return deepest

n=int(input())
if n==0:
    print(0)
    exit()
nodes=[]
for i in range(n):
    a,b=map(int,input().split())
    nodes.append((a,b))
deep=1
deepest=1
print(dfs(0))
```

##### 二叉树的遍历

```python
class TreeNode:
    def __init__(self,value):
        self.value=value
        self.left=None
        self.right=None

n=int(input())
nodes=[TreeNode(i+1) for i in range(n)]  #将整棵树储存在“节点“列表中，注意node()的写法
for i in range(n):
    l,r=map(int,input().split())
    if l!=-1:
        nodes[i].left=nodes[l-1]  #将列表的节点按照题目数据连接起来
    if r!=-1:
        nodes[i].right=nodes[r-1]

# 前序遍历二叉树
def preorderTraversal(root):
    # 递归法
    if not root: return []
    result = []

    def traversal(root):
        if not root:
            return
        result.append(root.value)  # 先将根节点值加入结果
        if root.left:
            traversal(root.left)  # 左
        if root.right:
            traversal(root.right)  # 右

    traversal(root)
    return result

def preorderTraversal2(root):
    # 迭代法
    if not root: return []
    stack = [root]
    res = []
    while stack:
        node = stack.pop()
        res.append(node.value)
        if node.right:  #先将右节点加入栈，出栈顺序才正确
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return res

#中序表达式遍历二叉树
#递归实现
def inorderTraversal(root):
    result = []

    def traversal(root):
        if root == None:
            return
        traversal(root.left)  # 左
        result.append(root.value)  # 中序
        traversal(root.right)  # 右

    traversal(root)
    return result

#迭代实现
def inorderTraversal2(root):
    if not root:
        return []  # 空树

    stack = []  # 不能提前将root结点加入stack中'
    res = []
    cur = root
    while cur or stack:
        if cur:  # 先迭代访问最底层左子树结点
            stack.append(cur)
            cur = cur.left
        else:  # 到达最左节点后处理栈顶结点
            cur = stack.pop()
            res.append(cur.value)
            cur = cur.right  # 取栈顶元素右结点
    return res

#后续表达式遍历
#递归实现
def postorderTraversal(root):
    # 递归遍历
    if not root:
        return []
    result = []

    def traversal2(root):
        if not root:
            return
        traversal2(root.left)  # 左
        traversal2(root.right)  # 右
        result.append(root.value)  # 中
    traversal2(root)
    return result

    #迭代实现
def postorderTraversal2(root):
    if not root: return []
    stack = [root]
    res = []
    while stack:
        node = stack.pop()
        res.append(node.value)
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    return res[::-1]
```

##### 解析树

```python
#处理括号树
def parse_tree(s):
    stack = []
    node = None
    for char in s:
        if char.isalpha():  # 如果是字母，创建新节点
            node = TreeNode(char)
            if stack:  # 如果栈不为空，把节点作为子节点加入到栈顶节点的子节点列表中
                stack[-1].children.append(node)
        elif char == '(':  # 遇到左括号，当前节点可能会有子节点
            if node:
                stack.append(node)  # 把当前节点推入栈中
                node = None
        elif char == ')':  # 遇到右括号，子节点列表结束
            if stack:
                node = stack.pop()  # 弹出当前节点
    return node 

#解析树扩展
class Stack(object):
    def __init__(self):
        self.items = []
        self.stack_size = 0

    def isEmpty(self):
        return self.stack_size == 0

    def push(self, new_item):
        self.items.append(new_item)
        self.stack_size += 1

    def pop(self):
        self.stack_size -= 1
        return self.items.pop()

    def peek(self):
        return self.items[self.stack_size - 1]

    def size(self):
        return self.stack_size
class BinaryTree:
    def __init__(self, rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None
    def insertLeft(self, newNode):
        if self.leftChild == None:
            self.leftChild = BinaryTree(newNode)
        else:  # 已经存在左子节点。此时，插入一个节点，并将已有的左子节点降一层。
            t = BinaryTree(newNode)
            t.leftChild = self.leftChild
            self.leftChild = t
    def insertRight(self, newNode):
        if self.rightChild == None:
            self.rightChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t
    def getRightChild(self):
        return self.rightChild
    def getLeftChild(self):
        return self.leftChild
    def setRootVal(self, obj):
        self.key = obj
    def getRootVal(self):
        return self.key
    def traversal(self, method="preorder"):
        if method == "preorder":
            print(self.key, end=" ")
        if self.leftChild != None:
            self.leftChild.traversal(method)
        if method == "inorder":
            print(self.key, end=" ")
        if self.rightChild != None:
            self.rightChild.traversal(method)
        if method == "postorder":
            print(self.key, end=" ")
def buildParseTree(fpexp):
    fplist = fpexp.split()
    pStack = Stack()
    eTree = BinaryTree('')
    pStack.push(eTree)
    currentTree = eTree
    for i in fplist:
        if i == '(':
            currentTree.insertLeft('')
            pStack.push(currentTree)
            currentTree = currentTree.getLeftChild()
        elif i not in '+-*/)':
            currentTree.setRootVal(int(i))
            parent = pStack.pop()
            currentTree = parent
        elif i in '+-*/':
            currentTree.setRootVal(i)
            currentTree.insertRight('')
            pStack.push(currentTree)
            currentTree = currentTree.getRightChild()
        elif i == ')':
            currentTree = pStack.pop()
        else:
            raise ValueError("Unknown Operator: " + i)
    return eTree
exp = "( ( 7 + 3 ) * ( 5 - 2 ) )"
pt = buildParseTree(exp)
for mode in ["preorder", "postorder", "inorder"]:
    pt.traversal(mode)
    print()

import operator  #
def evaluate(parseTree):
    opers = {'+':operator.add, '-':operator.sub, '*':operator.mul, '/':operator.truediv}
    leftC = parseTree.getLeftChild()
    rightC = parseTree.getRightChild()
    if leftC and rightC:
        fn = opers[parseTree.getRootVal()]
        return fn(evaluate(leftC),evaluate(rightC))
    else:
        return parseTree.getRootVal()
print(evaluate(pt))
# 30

#后序求值
def postordereval(tree):
    opers = {'+':operator.add, '-':operator.sub,
             '*':operator.mul, '/':operator.truediv}
    res1 = None
    res2 = None
    if tree:
        res1 = postordereval(tree.getLeftChild())
        res2 = postordereval(tree.getRightChild())
        if res1 and res2:
            return opers[tree.getRootVal()](res1,res2)
        else:
            return tree.getRootVal()

print(postordereval(pt))
# 30

#中序还原完全括号表达式
def printexp(tree):
    sVal = ""
    if tree:
        sVal = '(' + printexp(tree.getLeftChild())
        sVal = sVal + str(tree.getRootVal())
        sVal = sVal + printexp(tree.getRightChild()) + ')'
    return sVal

print(printexp(pt))
# (((7)+3)*((5)-2))
```

##### Huffman树和Huffman编码：

```python
class Node:
    def __init__(self, weight, char=None):
        self.weight = weight
        self.char = char
        self.left = None
        self.right = None
        
    def __lt__(self, other):
        return self.weight<other.weight #用于堆排序，不能省略
    
def build_huffman_tree(characters): # 输入为{node名称：权值……}
    heap = []
    for char, weight in characters.items():
        heapq.heappush(heap, Node(weight, char)) #在排序堆中加入character的所有节点
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(left.weight + right.weight)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged) #将建立的最小权值的子树作为节点，左右儿子的权值和作为新权值加入排序堆  #合并后char默认为None
    return heap[0]

def build_code(root):
    codes={}
    def traverse(node,code):
        if node.char: 			   #如果是叶节点（也可以用没有左右儿子判断）
            codes[node.char]=code  #在总编码codes中储存叶子名字对应的项目
        else:
            traverse(node.left,code+'0')
            traverse(node.right,code+'1')
    traverse(root,'') #从根节点开始遍历
    return codes

def encoding(codes,string):	  #将给叶子字符串序列输出为01编码
    encoded=''
    for char in string:
        encoded+=codes[char]
       return encoded

def decoding(root,encoded_string):  #将给定01编码转换为叶子序列
    decoded=''
    node=root
    for bit in encoded_string:
        if bit==0:
            node=node.left
        else:
            node=node.right
        if node.char:
            decoded+=node.char
            node=root
    return decoded

def wpl(node,depth=0):  #计算树的带权路径长度
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return depth*node.weight
    return (wpl(node.left,depth+1)+wpl(node.right,depth+1))

# 读取输入
n = int(input())
characters = {}
for _ in range(n):
    char, weight = input().split()
    characters[char] = int(weight)

huffman_tree = build_huffman_tree(characters)
codes = encode_huffman_tree(huffman_tree) #要先赋值给变量才能输出


```

##### 各序遍历建树

```python
def buildtree(preorder,inorder):
    if not preorder or not inorder:
        return None
    root=Node(preorder[0])
    rootindex=inorder.index(root.val)
    root.left=buildtree(preorder[1:rootindex+1],inorder[:rootindex])
    root.right=buildtree(preorder[rootindex+1:],inorder[rootindex+1:])
    return root
def build(postorder,inorder):
    if not postorder or not inorder:
        return None
    root_val=postorder[-1]
    root=node(root_val)
    mid=inorder.index(root_val)
    root.left=build(postorder[:mid],inorder[:mid])
    root.right=build(postorder[mid:-1],inorder[mid+1:])
    return root
```

##### 二叉搜索树

二叉搜索树在中序遍历时会得到递增序列

```python
#二叉搜索树的查找
def search_BST(root,value):
    if not root:
        return None
    
    if value == root.value:
        return root
    elif value < root.value:
        return search_BST(root.left,value)
    else:
        return search_BST(root.right,value)

#二叉搜索树的（末端）插入，以及用插入的方法建树
def insert_BST(root,value):
    if root == None:
        return TreeNode(value)
    
    if value < root.value:
        root.left = insert_BST(root.left,value)
    if value > root.value:
        root.right = insert_BST(root.right,value)
    return root

def build_BST(lst,rootvalue):
    root = TreeNode(rootvalue)
    for num in lst:
        if num != rootvalue:
            insert_BST(root,num)
    return root

#二叉搜索树的删除
def deleteNode(root, value) -> TreeNode:
    if not root:
        return root

    if root.value > value:
        root.left = deleteNode(root.left, value)
        return root
    elif root.value < value:
        root.right = deleteNode(root.right, value)
        return root
    else:
        if not root.left:
            return root.right
        elif not root.right:
            return root.left
        else:
            curr = root.right
            while curr.left:
                curr = curr.left
            curr.left = root.left
            return root.right
```

##### 平衡二叉树（AVL）

```python
class Node:
    def __init__(self,value):
        self.value=value
        self.left=None
        self.right=None
        self.height=1#树的高度
class AVL:
    '''二叉平衡搜索树'''
    def __init__(self):
        self.root=None#self.root表示AVL树根节点
    def insert(self,value):
        '''向avl树中插入值为value的元素'''
        if not self.root:#空树则创建树
            self.root=Node(value)
        else:
            self.root=self._insert(value,self.root)#非空则插入
    def _get_height(self, node):
        '''或许node的树高度'''
        if not node:
            return 0
        return node.height
    def _get_balance(self, node):
        '''获取节点node的平衡因子'''
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)
    
    def _rotate_left(self, z):
        '''左旋'''
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y

    def _rotate_right(self, y):
        '''右旋'''
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        return x
    def _insert(self,value,node):#value为新插入的点的值
        '_insert()函数的递归当中自底向上地检查祖先节点是否失衡'
        if not node:#空树则建树
            return Node(value)
        elif value<node.value:#应归于左子树
            node.left=self._insert(value,node.left)
        else:#应归于右子树
            node.right=self._insert(value, node.right)
            node.height=1+max(self._get_height(node.left),
                      self._get_height(node.right))
        #二叉树高度的递推式
        
        #调整平衡
        balance = self._get_balance(node)#当前节点的平衡因子
        
        if balance > 1:
            if value < node.left.value: # 树形是 LL
                return self._rotate_right(node)
            else:   # 树形是 LR
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)

        if balance < -1:
            if value > node.right.value:    # 树形是 RR
                return self._rotate_left(node)
            else:   # 树形是 RL
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)
        return node
    
    def preorder(self):
        '''输出前序遍历'''
        return self._preorder(self.root)

    def _preorder(self, node):
        if not node:
            return []
        return [node.value] + self._preorder(node.left) + self._preorder(node.right)

n = int(input().strip())
sequence = list(map(int, input().strip().split()))

avl = AVL()
for value in sequence:
    avl.insert(value)

print(' '.join(map(str, avl.preorder())))
```

##### 树的应用

```python
#烟花从侧面看
#通过深度遍历树（类似bfs）
#将树的父节点存入stack中，通过stack导出下一层的子节点，存入temp，一层结束后用temp覆盖stack
n = int(input())
tree = [0]
for i in range(n):
    tree.append(list(map(int, input().split())))
stack = [1]
ans = []
while stack:
    ans.append(str(stack[-1]))
    temp = []
    for x in stack:
        if tree[x][0] != -1:
            temp.append(tree[x][0])
        if tree[x][1] != -1:
            temp.append(tree[x][1])
    stack = temp
print(" ".join(ans))
```



#### 栈

```python
#合法出栈序列
# 或利用： 入栈的顺序是降序排列（如6,5,4,3,2,1)，由于栈是FILO，那么出栈序列任意数A的后面比A大的数都是按照升序排列的；入栈的顺序是升序排列（如1,2,3,4,5,6)，由于栈是FILO，那么出栈序列任意数A的后面比A小的数都是按照降序排列的。
def is_valid_pop_sequence(origin, output):
    if len(origin) != len(output):
        return False  # 长度不同，直接返回False

    stack = []
    bank = list(origin)
    
    for char in output:
        # 如果当前字符不在栈顶，且bank中还有字符，则继续入栈
        while (not stack or stack[-1] != char) and bank:
            stack.append(bank.pop(0))
        
        # 如果栈为空，或栈顶字符不匹配，则不是合法的出栈序列
        if not stack or stack[-1] != char:
            return False
        
        stack.pop()  # 匹配成功，弹出栈顶元素
    
    return True  # 所有字符都匹配成功

# 读取原始字符串
origin = input().strip()

# 循环读取每一行输出序列并判断
while True:
    try:
        output = input().strip()
        if is_valid_pop_sequence(origin, output):
            print('YES')
        else:
            print('NO')
    except EOFError:
        break
        
#单调栈
#记录i天以后的更高气温间隔
class DailyTemperature(object):
    def dailyTemperatures(self, temperatures: list) -> list:
        """
        :type temperatures: List[int]
        :rtype: List[int]
        """
        length = len(temperatures)
        if length <= 0:
            return []
        if length == 1:
            return [0]
        # res用来存放第i天的下一个更高温度出现在几天后
        res = [0] * length
        # 定义一个单调栈
        stack = []
        for index in range(length):
            current = temperatures[index]
            # 栈不为空 且 当前温度大于栈顶元素
            while stack and current > temperatures[stack[-1]]:
                # 出栈
                pre_index = stack.pop()
                # 当前索引和出栈索引差即为出栈索引结果
                res[pre_index] = index - pre_index
            stack.append(index)
        return res


if __name__ == "__main__":
    demo = DailyTemperature()
    temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
    print(demo.dailyTemperatures(temperatures)) 

#倒着检索：
def dayt(temp):
    n = len(temp)
    ans = [0] * n
    st = []
    for i in range(n-1,-1,-1):
        t = temp[i]
        while st and t >= temp[st[-1]]:
            st.pop()
        if st:
            ans[i] = st[-1] - i
        st.append(i)
    return ans

#奶牛排队
"""
简化题意：求一个区间，使得区间左端点最矮，区间右端点最高，且区间内不存在与两端相等高度的奶牛，输出这个区间的长度。
我们设左端点为 A ,右端点为 B
因为 A 是区间内最矮的，所以 [A.B]中，都比 A 高。所以只要 A 右侧第一个 ≤A的奶牛位于 B 的右侧，则 A 合法
同理，因为B是区间内最高的，所以 [A.B]中，都比 B 矮。所以只要 B 左侧第一个 ≥B 的奶牛位于 A的左侧，则 B合法
对于 “ 左/右侧第一个 ≥/≤ ” 我们可以使用单调栈维护。用单调栈预处理出 zz数组表示左，r 数组表示右。
然后枚举右端点 B寻找 A，更新 ans 即可。

这个算法的时间复杂度为 O(n)，其中 n 是奶牛的数量。
""""

N = int(input())
heights = [int(input()) for _ in range(N)]

left_bound = [-1] * N
right_bound = [N] * N

stack = []  # 单调栈，存储索引


# 求左侧第一个≥h[i]的奶牛位置
for i in range(N):
    while stack and heights[stack[-1]] < heights[i]:
        stack.pop()

    if stack:
        left_bound[i] = stack[-1]

    stack.append(i)

stack = []  # 清空栈以供寻找右边界使用

# 求右侧第一个≤h[i]的奶牛位
for i in range(N-1, -1, -1):
    while stack and heights[stack[-1]] > heights[i]:
        stack.pop()

    if stack:
        right_bound[i] = stack[-1]

    stack.append(i)

ans = 0

# for i in range(N-1, -1, -1):  # 从大到小枚举是个技巧
#     for j in range(left_bound[i] + 1, i):
#         if right_bound[j] > i:
#             ans = max(ans, i - j + 1)
#             break
#
#     if i <= ans:
#         break

for i in range(N):  # 枚举右端点 B寻找 A，更新 ans
    for j in range(left_bound[i] + 1, i):
        if right_bound[j] > i:
            ans = max(ans, i - j + 1)
            break
print(ans)
```



### 5、注意事项

1.边界值和特殊情况

2.整除和除余符号不搞混

3.dfs bfs等循环数多的，注意循环开头的几行代码

4.注意重新审题，题目中给出的限制

5.检查输入和输出格式是否完全一致，主要是输出

6.减少逻辑问题要在写的时候保持冷静