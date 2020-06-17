subg_fd = open('/home/lyt/static_analysis/result/graph_1.txt', "r")
line = subg_fd.readline()
    # dic is a dict like:
    # {1:{2,3,4},5:{6,7,8}}
    #which means op1's  predecessor is {2,3,4}
dic = dict()
while line:
    line = line.split('||')[0]
    split_line = line.split('\t')
    if len(split_line) > 1:
        curID = int(split_line[0].split('[')[1].split(']')[0])
        for ID in split_line[1:]:
            if int(ID) not in dic.keys():
                dic[int(ID)] = set()
            dic[int(ID)].add(curID)
    line = subg_fd.readline()
dic[0] = set()
subg_fd.close()


path = list()
filereader = open('result_by_graph_1_minibatch32_299', 'r')
line = filereader.readline()
numnum = 0
while line:
    #print(line.split("[")[1])
    if len(line)>1:
        split_line = line.split("[")
        split_line = split_line[1]
        cur = split_line.split("]")[0]
        path.append(int(cur))
    line = filereader.readline()
filereader.close()
pathset = set(path)


def get_subgraph(sink):
    queue = list()
    result = set()
    queue.append(sink)
    while len(queue) > 0:
        cur = queue[0]
        result.add(cur)
        queue.remove(queue[0])
        if cur in dic.keys():
            for predecessor in dic[cur]:
                if predecessor not in pathset:
                    queue.append(predecessor)
    return result

allpre_10551 = dict()
for pre in dic[10551]:
    allpre_10551[pre] = get_subgraph(pre)
print(allpre_10551[6748])

subg_fd = open('/home/lyt/static_analysis/result/graph_1.txt', "r")
op_information = subg_fd.readlines()
subg_fd.close()

node_num = len(op_information)
uf = list()
def myfind(uf, a):
    if uf[a] != a:
        uf[a] = myfind(uf, uf[a])
    return uf[a]
def myunion(uf, a, b):
    fathera = myfind(uf, a)
    fatherb = myfind(uf, b)
    if fathera != fatherb:
        uf[fatherb] = fathera
#init unionfind
for i in range(node_num):
  uf.append(i)

for i in range(node_num):
  curline = op_information[i]
  curdata = curline.split('||')[1].strip(' ')
  if(curdata != '\n'):
    curdata = curdata.split(' ')
    for pair in curdata[:-1]:
        dtype = pair.split(',')[0].split('{')[1]
        successor = pair.split(',')[1].split('}')[0]
        if int(dtype) >= 100:
            myunion(uf, i,int(successor))

allroot = set()
for i in range(len(uf)):
    allroot.add(myfind(uf, i))
pathinuf = set() #这个集合记录了critical path中的节点所在的uf根节点
for node in path:
  pathinuf.add(myfind(uf, node))

  #读取init图记录为map
graph_0 = open('/home/lyt/static_analysis/result/graph_0.txt', "r")
line = graph_0.readline()
graph_0_map = dict()
#{name: id,}
while line:
    line = line.split('[')
    cur_id = int(line[1].split(']')[0])
    cur_name = line[3].split(']')[0]
    graph_0_map[cur_name] = cur_id
    line = graph_0.readline()
graph_0.close()
#读取init图记录为并查集
graph_0 = open('/home/lyt/static_analysis/result/graph_0.txt', "r")
graph_0_information = graph_0.readlines()
graph_0.close()
node_num = len(graph_0_information)
graph_0_uf = list()
for i in range(node_num):
    graph_0_uf.append(i)

for i in range(node_num):
  curline = graph_0_information[i]
  curdata = curline.split('||')[1].strip(' ')
  if(curdata != '\n'):
    curdata = curdata.split(' ')
    for pair in curdata[:-1]:
        dtype = pair.split(',')[0].split('{')[1]
        successor = pair.split(',')[1].split('}')[0]
        if int(dtype) >= 100:
            myunion(graph_0_uf, i,int(successor))

replacement_list = allpre_10551[9708]
#replacement_list = replacement_list  | allpre_10551[7664]
#replacement_list = replacement_list  | allpre_10551[9708]
#replacement_list = replacement_list  | allpre_10551[10537]
#replacement_list = replacement_list  | allpre_10551[6802]
#replacement_list = replacement_list  | allpre_10551[10524]
#replacement_list = replacement_list  | allpre_10551[6830]
#replacement_list = replacement_list  | allpre_10551[10511]
#replacement_list = replacement_list  | allpre_10551[6857]
#replacement_list = replacement_list  | allpre_10551[10498]
'''
import random
for x in allpre_10551:
    rand = random.random()
    if rand < 0.3:
        replacement_list = replacement_list | allpre_10551[x]
'''
replacement_remove = set()
for node in replacement_list:
    if myfind(uf, node) in pathinuf:
        replacement_remove.add(node)
replacement_list = replacement_list - replacement_remove
replacement_add = set()
for node in replacement_list:
    for x in range(len(uf)):
        if myfind(uf, x) == myfind(uf, node):
            replacement_add.add(x)
replacement_list = replacement_list | replacement_add
gpu_id = 1
fileWriter = open('/home/lyt/static_analysis/replacement/replace_train.txt', 'a')
graph_0_replacement_list = set()
for x in replacement_list:
    data = op_information[x].split('[')[3]
    data = data.split(']')[0]
    #添加graph_0需要replace的节点
    if str(data) in graph_0_map.keys():
        graph_0_replacement_list.add(graph_0_map[str(data)])
    fileWriter.write(str(data) + ' ' + str(gpu_id) + '\n')

fileWriter.close()

graph_0_replacement_add = set()
for node in graph_0_replacement_list:
    for x in range(len(graph_0_uf)):
        if myfind(graph_0_uf, x) == myfind(graph_0_uf, node):
            graph_0_replacement_add.add(x)
graph_0_replacement_list = graph_0_replacement_list | graph_0_replacement_add
fileWriter = open('/home/lyt/static_analysis/replacement/replace_init.txt', 'a')
for x in graph_0_replacement_list:
    data = graph_0_information[x].split('[')[3]
    data = data.split(']')[0]
    fileWriter.write(str(data) + ' ' + str(gpu_id) + '\n')
fileWriter.close()
