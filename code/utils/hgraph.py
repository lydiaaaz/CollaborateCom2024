import dgl
from collections import defaultdict
import torch
from copy import deepcopy

class HGraph:  # 对图的高层抽象，包含多个级联（Cascade）和用户之间的关系
    def __init__(self, num_user, num_cas):
        self.num_user = num_user
        self.num_cas = num_cas
        self.init()

    # 向图中插入级联，更新用户之间的关系、级联信息等。这个方法在处理传入的级联数据时，更新了用户之间的关注和被关注关系，并将级联信息插入到相应的级联对象中。
    def insert(self, cascades, srcs, dsts, abs_times, pub_times):
        self.cas_batch = defaultdict(list)
        times = abs_times - pub_times
        for cas, src, dst, time, abs_time in zip(cascades, srcs, dsts, times, abs_times):
            self.user_neighbor['follower'][src].append(dst)
            self.user_time['follower'][src].append(abs_time)
            self.user_neighbor['follow'][dst].append(src)
            self.user_time['follow'][dst].append(abs_time)
            self.cascades[cas].insert(src, dst, time, abs_time)
            self.cas_batch[cas].append((src, dst))

    def get_cas_seq(self, cascades):  # 获取所有级联的节点序列
        users, times, valid_length = [], [], []
        for cas in cascades:
            user, time = self.cascades[cas].get_seq()
            users.append(torch.tensor(user, dtype=torch.long))
            times.append(torch.tensor(time))
            valid_length.append(len(user))
        return users, times, valid_length

    def get_cas_pub_time(self, cascades):  # 获取所有级联的发布时间
        cas_pub_times = []
        for cas in cascades:
            cas_pub_times.append(self.cascades[cas].get_pub_time())
        return cas_pub_times

    def get_cas_graph(self, cascades):  # 获取所有级联的图表示
        graph_roots, graph_leafs = [], []
        for cas in cascades:
            graph_root, graph_leaf = self.cascades[cas].get_graph()
            graph_roots.append(graph_root)
            graph_leafs.append(graph_leaf)
        return dgl.batch(graph_roots), dgl.batch(graph_leafs)  # dgl.batch将图列表（graphs）按照批量处理的方式合并成一个大图（batched graph）

    def batch_cas_info(self): # 返回一个深拷贝的级联信息字典
        return deepcopy(self.cas_batch)

    def init(self):  # 初始化图的一些属性，如用户邻居关系、级联对象等
        self.user_neighbor = {'follow': defaultdict(list),
                              'follower': defaultdict(list),
                              'neighbor': defaultdict(list)}
        self.user_time = deepcopy(self.user_neighbor)
        self.cascades = defaultdict(lambda: Cascade())
        # 创建一个默认值为Cascade()的字典（键的类型实际上是级联的标识符），当尝试访问一个不存在的键时，会自动调用Cascade()构造函数生成一个新的Cascade对象作为默认值。lambda: Cascade() 是一个匿名函数，它没有函数名，冒号后是函数体。


class Cascade:  # 表示图中的级联结构
    def __init__(self):
        self.seq = []  # 节点序列(v, t)
        self.dag = []  # 有向边的信息(u, v, t)
        self.pub_time = 1000000  # 发布时间
        self.cnt = 0
        self.node2id = dict()  # 用于将节点映射到整数标识符的字典
        self.id2node = dict()  # 用于将节点的整数标识符映射回节点本身的字典
        self.node_times = []

    def insert(self, u, v, t, abs_time):  # 向级联中插入新的节点和边，同时更新发布时间。此外，该方法还构建了节点的编号映射关系，以及节点的时间信息。
        self.seq.append((v, t))
        self.pub_time = min(self.pub_time, abs_time)
        if v in self.node2id:  # 如果节点 v 存在于 self.node2id 字典中，说明它之前已经在级联中出现过，可以继续进行下一步的判断
            if u not in self.node2id:
                return  # 如果节点 u 不在 self.node2id 字典中，这可能意味着该节点是一个新的节点，需要添加到字典中
            if self.node_times[self.node2id[u]] >= self.node_times[self.node2id[v]]:
                return  # 如果节点 u 不是新节点，并且它的时间戳大于等于节点 v 的时间戳，则提前结束当前函数的执行并返回，不进行后续的节点插入操作
        for x in [u, v]:
            if x not in self.node2id:  # 对于每个节点 x，如果它之前没有在 node2id 中出现，就将其添加到 node2id 中
                self.node2id[x] = self.cnt  # 当新节点被插入时，会更新 node2id 字典，将节点映射到相应的整数标识符
                self.id2node[self.cnt] = x  # 当新节点被插入时，会更新 id2node 字典，将节点的整数标识符与节点本身建立映射关系
                self.cnt += 1
                self.node_times.append(t)
        self.dag.append((u, v, t))

    def get_graph(self):  # 获取级联的有向图表示，包括根节点和叶子节点的图。通过构建节点映射关系和时间信息，将有向边的信息转换成DGL图的表示，并进行一些处理，如去除自环和添加节点的掩码信息。
        srcs, dsts, times = zip(*self.dag)
        nodes = list(set(srcs) | set(dsts))
        ids = list(range(len(nodes)))
        map_srcs = list(map(lambda x: self.node2id[x], srcs))  # 源节点的列表，表示有向边中的起始节点
        map_dsts = list(map(lambda x: self.node2id[x], dsts))  # 目标节点的列表，表示有向边中的结束节点
        graph_leaf = dgl.graph((map_srcs, map_dsts))  # 创建一个有向图对象，map_srcs和map_dsts这两个列表中的元素一一对应，构成了有向图中的边
        # ndata 和 edata 分别表示节点数据（node data）和边数据（edge data）的属性
        graph_leaf.ndata['id'] = torch.tensor(list(map(lambda x: self.id2node[x], ids)))
        # list(map(lambda x: self.id2node[x], ids))：使用 map 函数将节点的整数标识符（ids）映射为相应的节点标识符（self.id2node[x]），其中 x 是节点的整数标识符。
        # torch.tensor(...)：将映射得到的节点标识符列表转换为一个 PyTorch 张量。
        # graph_leaf.ndata['id'] = ...：将上述生成的节点标识符张量赋值给节点数据中的 'id' 属性。
        # 因此，graph_leaf.ndata['id'] 包含了图中每个节点的标识符信息。
        graph_leaf.edata['time'] = torch.tensor(times, dtype=torch.float)  # 在边数据（edge data）中添加一个名为 'time' 的属性，该属性包含了与每条边相关的时间信息
        # times 是一个包含了每条边对应的时间信息的列表。
        # torch.tensor(times, dtype=torch.float) 将时间信息列表转换为一个 PyTorch 张量，并指定数据类型为浮点数。
        # graph_leaf.edata['time'] 将上述生成的时间信息张量赋值给边数据中的 'time' 属性。
        # 因此，graph_leaf.edata['time'] 包含了图中每条边的时间信息。
        graph_leaf = dgl.RemoveSelfLoop()(graph_leaf) # 去除图中的自环
        graph_leaf.ndata['mask'] = torch.tensor(graph_leaf.out_degrees() == 0, dtype=torch.float).unsqueeze(dim=-1) # 创建了一个节点掩码张量，添加节点的掩码信息，过滤掉出度为0的节点（掩码值为1，否则为0）
        # graph_leaf.out_degrees() == 0 表示对图中每个节点计算其出度，然后判断是否等于0。这个条件产生一个布尔值的张量，表示每个节点是否具有出边。
        # torch.tensor(..., dtype=torch.float) 将布尔值的张量转换为浮点数类型的张量，其中1表示条件为真，0表示条件为假。
        # .unsqueeze(dim=-1) 在张量的最后一维度上增加一个维度，使得结果成为形状为 (节点数, 1) 的张量。
        # 最终，这个新创建的节点数据属性 'mask' 包含了关于每个节点是否具有出边的信息。

        graph_root = dgl.reverse(graph_leaf, share_ndata=True, share_edata=True) # 每条边的方向都被反转
        graph_root.ndata['mask'] = torch.tensor(graph_root.out_degrees() == 0, dtype=torch.float).unsqueeze(dim=-1)

        return graph_root, graph_leaf

    def get_seq(self): # 返回级联中节点的序列和对应的时间信息
        users, times = zip(*self.seq)
        return users, times

    def get_pub_time(self): # 返回级联的发布时间
        return self.pub_time
