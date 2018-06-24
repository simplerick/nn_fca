from fca import*
from heapq import*

fca = FCA()
fca.load_lattice()
fca.load_properties()



def order1(i,a1,a2,a3,remain):
    if len(remain) != 0:
        cover = len(set(fca.lattice[i]['extent'])&(remain))/len(remain)
    else:
        cover = 1
    return -(a1*fca.stability[i]+a2*fca.purity[i]+a3*fca.support[i]+cover)



def select_good(a1,a2,a3,size):
    concepts = []
    selected = []
    remain = set(fca.G)

    for i in range(len(fca.lattice)):
        heappush(concepts,[order1(i,a1,a2,a3,remain),i])

    for _ in range(size):
        # print(len(remain))
        o,c = heappop(concepts)
        o = order1(c,a1,a2,a3,remain)
        if o >  concepts[0][1]:
            heappush(concepts,[o,c])
        else:
            selected.append(c)
            remain -= set(fca.lattice[c]['extent'])

    return frozenset(selected)


selected = set()
params_values = {}
configs = []

A1 = np.arange(1,1.5,0.5)
A2 = np.arange(1,1.5,0.5)
A3 = np.arange(1,1.5,0.5)
min_size = 30
max_size = 30
Size = np.arange(min_size,max_size+1,1)

for size in Size:
    print("%d %%" % ((size-min_size+1)*100/(max_size-min_size+1)))
    for a1 in A1:
        for a2 in A2:
            for a3 in A3:
                s = select_good(a1,a2,a3,size)
                if s in selected:
                    params_values[s].append((a1,a2,a3,size))
                else:
                    selected.add(s)
                    params_values[s] = [(a1,a2,a3,size)]


print(len(selected))

for sample in selected:
    fca.load_lattice()
    fca.load_properties()
    fca.reduce_lattice(sample)
    adj = fca.build_cover_relation(reverse=True)
    conf = fca.calculate_conf()
    res_connect = {}
    for i,c in fca.partition_to_classes.items():
        if c not in res_connect:
            res_connect[c] = []
        res_connect[c].append(i)
    weights = {}
    for i in adj:
        if len(adj[i]) == 0:
            weights[i] = fca.lattice[i]['intent']
    configs.append((adj,res_connect,weights, conf))
    params_values[str(adj)] = params_values.pop(sample)

# print(configs)
np.save("data/configs", configs)
np.save("data/params_values", params_values)




# stability = list(fca.stability.items())
# support = list(fca.support.items())
# purity = list(fca.purity.items())

# stability.sort(key=lambda x: x[1],  reverse = True)
# stability = np.array(stability, dtype=[('id', 'int'), ('value', 'float')])


# support.sort(key=lambda x: x[1],  reverse = True)
# support = np.array(support, dtype=[('id', 'int'), ('value', 'float')])


# purity.sort(key=lambda x: x[1],  reverse = True)
# purity = np.array(purity, dtype=[('id', 'int'), ('value', 'float')])

# configurations = set()
# num = 100 # first 'num' concepts will be searched
# for i in range(num):
#     print("%d %%" % i)
#     if support[i][0] == support[i+1][0]:
#         continue
#     for j in range(num):
#         if purity[j][0] == purity[j+1][0]:
#             continue
#         for k in range(num):
#             if stability[k][0] == stability[k+1][0]:
#                 continue
#             concepts = set(np.concatenate((purity['id'][:i+1],purity['id'][:j+1],stability['id'][:k+1])))
#             if len(concepts) > 15:
#                 continue

#             fca.load_lattice()
#             fca.load_properties()
#             fca.reduce_lattice(concepts)
#             cover = set()
#             for concept in fca.lattice:
#                 cover |= set(concept['extent'])
#             adj = fca.build_cover_relation(reverse=True)
#             diversity = len(cover)/len(fca.G)
#             # print(diversity)
#             configurations.add((diversity,str(adj)))

# print(len(configurations))
# configurations = sorted(list(configurations),key=lambda x: x[0], reverse=True)
# print(configurations[:20])
# np.save("configurations", list(map(eval,configurations)))