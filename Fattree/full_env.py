import heapq
import threading
from enum import Enum
import random
import queue

import numpy as np
import networkx as nx


class EvType(Enum):
    SELF = 1
    STREAM = 2
    REMOTE = 3


class Event:
    def __init__(self,
                 time: float,
                 ev_type: EvType,
                 code: int,
                 data,
                 module,
                 ):
        self.time = time
        self.type = ev_type
        self.code = code
        self.data = data
        self.module = module

    def __lt__(self, other):
        return self.time < other.time


class Engine:
    def __init__(self):
        self.time = 0
        # q是优先队列
        self.q = []

    def sim_time(self):
        return self.time

    def add_event(self, event):
        heapq.heappush(self.q, event)

    def clear_events(self):
        self.q = []

    def run(self):
        while len(self.q) > 0 and self.time < 300:
            ev = heapq.heappop(self.q)
            self.time = ev.time
            # 执行事件
            ev.module.execute(ev)


class Component:
    def __init__(self, engine: Engine):
        self.engine = engine


class MatchType(Enum):
    SRC_ADDR = 1
    DST_ADDR = 2
    SRC_PORT = 3
    DST_PORT = 4
    PROTOCOL = 5


class Match:
    def __init__(self, match_type: MatchType, value):
        self.type = match_type
        self.value = value


class ActionType(Enum):
    OUTPUT = 1
    DROP = 2


class Action:
    def __init__(self, action_type: ActionType, value):
        self.type = action_type
        self.value = value


class Packet:
    def __init__(self, size):
        self.protocol = None
        self.src_port = None
        self.dst_port = None
        self.dst_addr = None
        self.src_addr = None
        self.size = size

    # 返回数据包的长度，单位是字节
    def __len__(self):
        return self.size


class OFPacketType(Enum):
    PACKET_IN = 1
    PACKET_OUT = 2
    FLOW_MOD = 3


class OFPacket(Packet):
    def __init__(self, dp, size, pkt_type: OFPacketType):
        super().__init__(size)
        self.dp = dp
        self.protocol = 'OpenFlow'
        self.type = pkt_type
        self.data_pkt = None


class DataPacket(Packet):
    def __init__(self, size, flow_id):
        super().__init__(size)
        self.protocol = 'Data'
        self.flow_id = flow_id
        self.last_pkt = False


class FlowEntry:
    def __init__(self, matches, actions, priority=0):
        self.matches = matches
        self.actions = actions
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority

    def execute(self, packet):
        # 执行动作
        for action in self.actions:
            if action.type == ActionType.OUTPUT:
                # 发送到交换机
                return action.value
            elif action.type == ActionType.DROP:
                # 丢弃
                return None

    def match(self, packet: Packet):
        # 匹配数据包
        for match in self.matches:
            if match.type == MatchType.SRC_ADDR:
                if packet.src_addr != match.value:
                    return False
            elif match.type == MatchType.DST_ADDR:
                if packet.dst_addr != match.value:
                    return False
            elif match.type == MatchType.SRC_PORT:
                if packet.src_port != match.value:
                    return False
            elif match.type == MatchType.DST_PORT:
                if packet.dst_port != match.value:
                    return False
            elif match.type == MatchType.PROTOCOL:
                if packet.protocol != match.value:
                    return False
        return True


# engine = Engine()


class FlowModType(Enum):
    ADD = 1
    DELETE = 2


class Controller(Component):
    def __init__(self, engine: Engine, G):
        super().__init__(engine)
        self.G = G

        self.processed_flows = set()

    def execute(self, ev: Event):
        if ev.type == EvType.REMOTE:
            # 控制平面事件
            if not isinstance(ev.data, OFPacket):
                raise Exception('Invalid OF packet type')
            if ev.data.type == OFPacketType.PACKET_IN:
                # 处理数据包
                self.handle_packet_in(ev.data.dp, ev.data.data_pkt)

    def handle_packet_in(self, dp, packet: DataPacket):
        # 不重复处理流的请求，故障了就让这条流无法发送到
        if packet.flow_id in self.processed_flows:
            return
        self.processed_flows.add(packet.flow_id)

        # print('handle packet in')
        # 根据数据包的源节点和目的节点，从G中找到所有的最短路径
        paths = nx.all_shortest_paths(self.G, dp.dp_id, packet.dst_addr)
        paths = list(paths)

        # 聚合统计信息

        # Option 1: 只用前8个节点
        # node_keys = ['core_{}'.format(i) for i in range(4)] + \
        #             ['agg_0_{}'.format(i) for i in range(2)] + \
        #             ['edge_0_{}'.format(i) for i in range(2)]

        # Option 2: 用所有节点
        node_keys = ['core_{}'.format(i) for i in range(4)]
        for pod in range(4):
            for i in range(2):
                node_keys.append('agg_{}_{}'.format(pod, i))
            for i in range(2):
                node_keys.append('edge_{}_{}'.format(pod, i))

        states = []
        for node_key in node_keys:
            node = self.G.nodes[node_key]['instance']
            # states.append([
            #     np.sum(node.stats['recv_pkts']) / 20,
            #     np.sum(node.stats['recv_bytes']) / 10000,
            #     np.sum(node.stats['fwd_pkts']) / 20,
            #     np.sum(node.stats['fwd_bytes']) / 10000,
            # ])
            # todo: fixme disable stats
            states.append([
                np.sum(node.stats['recv_pkts']) / 10,
                np.sum(node.stats['recv_bytes']) / 10000,
                np.sum(node.stats['fwd_pkts']) / 10,
                np.sum(node.stats['fwd_bytes']) / 10000,
            ])
        # print(states)

        # 找到host节点对应的邻居
        for key, value in self.G[packet.src_addr].items():
            src_edge_sw = key
        for key, value in self.G[packet.dst_addr].items():
            dst_edge_sw = key

        available_cores = []
        for p in paths:
            for node in p:
                if 'core_' in node:
                    # 取出core节点的数字
                    core_id = int(node.split('_')[1])
                    available_cores.append(core_id)

        # 与AI联动，由AI选择最优路径
        self.engine.req_queue.put((states,
                                   packet.src_port,
                                   self.G.nodes[src_edge_sw]['id'],
                                   self.G.nodes[dst_edge_sw]['id'],
                                   available_cores
                                   ))

        res = self.engine.res_queue.get(block=True)
        if res == 'exit':
            # 直接清空事件队列，结束仿真
            self.engine.clear_events()
            return
        else:
            path = None
            # 找到包含core_{res}的路径
            for p in paths:
                if 'core_%s' % res in p:
                    path = p
                    break
            if path is None:
                return

        # 随机选择一条最短路径
        # path = random.choice(list(paths))
        # path = list(paths)[0]

        # 循环遍历路径上的交换机，下发流表
        for i in range(len(path) - 1):
            src_node = self.G.nodes[path[i]]['instance']
            edge = self.G[path[i]][path[i + 1]]
            matches = [Match(MatchType.SRC_ADDR, packet.src_addr),
                       Match(MatchType.DST_ADDR, packet.dst_addr),
                       Match(MatchType.SRC_PORT, packet.src_port),
                       Match(MatchType.DST_PORT, packet.dst_port), ]

            output_port = None
            if edge['port1'].sw == src_node:
                output_port = edge['port1'].port_no
            elif edge['port2'].sw == src_node:
                output_port = edge['port2'].port_no

            if output_port is None:
                raise Exception('No valid output port')

            actions = [Action(ActionType.OUTPUT, output_port)]
            flow_entry = FlowEntry(matches, actions)

            pkt = OFPacket(src_node, 0, OFPacketType.FLOW_MOD)
            pkt.flow_mod_type = FlowModType.ADD
            pkt.flow_entry = flow_entry

            # 下发流表
            self.send_packet_to_sw(src_node, pkt)

    def send_packet_to_sw(self, dp, packet):
        # 控制平面延迟，单位是秒，这里是1ms
        delay = 0.001
        # 添加Event
        ev = Event(self.engine.sim_time() + delay, EvType.REMOTE, 0, packet, dp)
        self.engine.add_event(ev)


# 交换机端口
class OFPort(Component):
    def __init__(self, engine, sw, port_no):
        super().__init__(engine)
        self.speed = 1_000_000  # bps
        self.port_no = port_no
        self.sw = sw
        self.peer = None

    def send(self, packet: Packet):
        # 根据数据包大小计算延迟，单位是秒
        delay = len(packet) * 8 / self.speed
        # 添加Event
        ev = Event(self.engine.sim_time() + delay, EvType.STREAM, 0, packet, self.peer.sw)
        self.engine.add_event(ev)


# 仿真OpenFlow交换机
class Switch(Component):
    INTRPT_CODE_SHOULD_PROCESS_PKT = 1
    INTRPT_CODE_COLLECT_STATS = 2

    COLLECT_INTERVAL = 0.1  # 采样间隔
    WINDOW_SIZE = 10  # 采样窗口大小

    def __init__(self, engine: Engine, dp_id, controller):
        super().__init__(engine)
        self.process_speed = 20000  # 每秒处理的数据包数量
        self.flow_table = []
        self.ports = dict()
        self.dp_id = dp_id
        self.controller = controller

        self.q = []
        self.is_busy = False
        self.pending_pkt = None

        self.stats = {
            "recv_pkts": [],
            "recv_bytes": [],
            "fwd_pkts": [],
            "fwd_bytes": [],
        }
        self.recv_pkts = 0
        self.recv_bytes = 0
        self.fwd_pkts = 0
        self.fwd_bytes = 0

        # 设置自中断
        ev = Event(self.engine.sim_time() + self.COLLECT_INTERVAL, EvType.SELF, self.INTRPT_CODE_COLLECT_STATS, None,
                   self)
        self.engine.add_event(ev)

    def add_port(self):
        port = OFPort(self.engine, self, len(self.ports))
        self.ports[port.port_no] = port
        return port

    def execute(self, ev: Event):
        # 处理事件
        if ev.type == EvType.STREAM:
            # 记录统计信息
            self.recv_pkts += 1
            self.recv_bytes += len(ev.data)

            if self.is_busy:
                # 交换机忙，将数据包放入队列
                self.q.append(ev.data)
            else:
                # 交换机空闲，处理数据包
                self.is_busy = True
                # 计算处理时间
                delay = 1 / self.process_speed
                self.pending_pkt = ev.data
                # 添加Event
                ev = Event(self.engine.sim_time() + delay, EvType.SELF, self.INTRPT_CODE_SHOULD_PROCESS_PKT, None, self)
                self.engine.add_event(ev)

        elif ev.type == EvType.SELF:
            # 处理自中断
            if ev.code == self.INTRPT_CODE_SHOULD_PROCESS_PKT:
                pkt = self.pending_pkt
                found = False
                # 遍历流表
                for flow in self.flow_table:
                    if flow.match(pkt):
                        found = True
                        outport = flow.execute(ev.data)
                        if outport is not None:
                            # 转发到对应的端口去
                            # todo: 暂时先不考虑FLOOD等特殊端口
                            port = self.ports[outport]
                            if port is not None:
                                # 记录统计信息
                                self.fwd_pkts += 1
                                self.fwd_bytes += len(pkt)

                                # 转发数据包
                                port.send(pkt)
                        break
                if not found:
                    # 发生了table-miss
                    # print('[%f]table-miss' % self.engine.sim_time(), self.dp_id, pkt)
                    self.send_packet_in(pkt)

                # 处理完毕, 处理队列中的数据包
                if len(self.q) > 0:
                    # 从队列去取出数据包开始处理
                    self.is_busy = True
                    # 计算处理时间
                    delay = 1 / self.process_speed
                    self.pending_pkt = self.q.pop(0)
                    # 添加Event
                    ev = Event(self.engine.sim_time() + delay, EvType.SELF, self.INTRPT_CODE_SHOULD_PROCESS_PKT, None,
                               self)
                    self.engine.add_event(ev)
                else:
                    # 队列为空，交换机空闲
                    self.is_busy = False
                    self.pending_pkt = None
            elif ev.code == self.INTRPT_CODE_COLLECT_STATS:

                if len(self.stats["recv_pkts"]) == self.WINDOW_SIZE:
                    self.stats["recv_pkts"].pop()
                if len(self.stats["recv_bytes"]) == self.WINDOW_SIZE:
                    self.stats["recv_bytes"].pop()
                if len(self.stats["fwd_pkts"]) == self.WINDOW_SIZE:
                    self.stats["fwd_pkts"].pop()
                if len(self.stats["fwd_bytes"]) == self.WINDOW_SIZE:
                    self.stats["fwd_bytes"].pop()

                self.stats["recv_pkts"].insert(0, self.recv_pkts)
                self.stats["recv_bytes"].insert(0, self.recv_bytes)
                self.stats["fwd_pkts"].insert(0, self.fwd_pkts)
                self.stats["fwd_bytes"].insert(0, self.fwd_bytes)

                # 清空统计数据
                self.recv_pkts = 0
                self.recv_bytes = 0
                self.fwd_pkts = 0
                self.fwd_bytes = 0

                # 设置自中断
                ev = Event(self.engine.sim_time() + self.COLLECT_INTERVAL, EvType.SELF, self.INTRPT_CODE_COLLECT_STATS,
                           None, self)
                self.engine.add_event(ev)

        elif ev.type == EvType.REMOTE:
            # 来自控制器的数据包
            pkt = ev.data
            # 判断数据包是否为OpenFlow数据包
            if not (pkt.protocol == 'OpenFlow' and isinstance(pkt, OFPacket)):
                raise Exception('Not OpenFlow packet')
            if pkt.type == OFPacketType.PACKET_OUT:
                # PACKET_OUT 数据包
                # todo:
                pass
            elif pkt.type == OFPacketType.FLOW_MOD:
                # 流表变更
                if pkt.flow_mod_type == FlowModType.ADD:
                    # 添加流表
                    heapq.heappush(self.flow_table, pkt.flow_entry)
                # todo:
                # print('[%f]flow_mod' % self.engine.sim_time(), self.dp_id, pkt.flow_mod_type, pkt.flow_entry)
            else:
                raise Exception('Unsupported OpenFlow packet type')

    def send_packet_to_controller(self, packet):
        delay = 0.001
        ev = Event(self.engine.sim_time() + delay, EvType.REMOTE, 0, packet, self.controller)
        self.engine.add_event(ev)

    def send_packet_in(self, data_pkt):
        # 创建packet-in数据包
        pkt = OFPacket(self, 0, OFPacketType.PACKET_IN)
        pkt.data_pkt = data_pkt

        # 发送数据包
        self.send_packet_to_controller(pkt)


# 仿真主机
class Host(Component):
    def __init__(self, engine: Engine, addr, flow_list):
        super().__init__(engine)
        self.addr = addr
        self.dp_id = addr
        self.port = None
        self.flow_list = flow_list

        for i in range(len(self.flow_list)):
            if flow_list[i]['src_ip'] == self.addr:
                engine.add_event(Event(self.flow_list[i]['start_time'], EvType.SELF, 0, i, self))

    def add_port(self):
        port = OFPort(self.engine, self, 0)
        self.port = port
        return port

    def execute(self, ev: Event):
        # 处理事件
        if ev.type == EvType.STREAM:
            pkt = ev.data
            # 收到数据包
            # print('[%f] host %s recv pkt' % (self.engine.sim_time(), self.dp_id), pkt)
            if pkt.last_pkt:
                fct = self.engine.sim_time() - self.flow_list[pkt.flow_id]['start_time']
                self.flow_list[pkt.flow_id]['fct'] = fct
                # 最后一个数据包，流结束
                # print('[%f] flow %d finished, FCT: %f' % (self.engine.sim_time(), pkt.flow_id, fct))
        elif ev.type == EvType.SELF:
            flow_id = ev.data

            # 处理自中断， 准备发送数据包
            flow = self.flow_list[flow_id]

            is_first = False
            if flow['remain_size'] == flow['size']:
                is_first = True

            pk_size = 0
            if flow['remain_size'] > 1400:
                pk_size = 1400
                flow['remain_size'] -= 1400
            else:
                pk_size = flow['remain_size']
                flow['remain_size'] = 0

            # 创建数据包
            pkt = DataPacket(pk_size, flow_id)
            pkt.src_addr = self.addr
            pkt.dst_addr = flow['dst_ip']
            pkt.src_port = flow['src_port']
            pkt.dst_port = flow['dst_port']
            pkt.last_pkt = flow['remain_size'] == 0

            if self.port is None:
                raise Exception('Port not set')
            # print('[%f]Host %s send flow %d' % (self.engine.sim_time(), self.addr, flow_id))
            self.port.send(pkt)

            if flow['remain_size'] > 0:
                # 还有数据包需要发送
                if is_first:
                    delay = 0.5
                else:
                    delay = self.get_flow_interval(flow_id)

                # 添加Event
                ev = Event(self.engine.sim_time() + delay, EvType.SELF, 0, flow_id, self)
                self.engine.add_event(ev)

    def get_flow_interval(self, flow_id):
        flow = self.flow_list[flow_id]
        if flow['size'] < 15000:
            # 小流, 0.25+均匀分布0.1
            return 0.25 + random.random() * 0.1
        else:
            # 大流, 0.05+均匀分布0.1
            return 0.05 + random.random() * 0.1


def connect_node(G: nx.Graph, sw1, sw2):
    port1 = sw1.add_port()
    port2 = sw2.add_port()
    port1.peer = port2
    port2.peer = port1
    G.add_edge(sw1.dp_id, sw2.dp_id, port1=port1, port2=port2)


def build_fattree_network(engine: Engine, flow_list):
    G = nx.Graph()

    c = Controller(engine, G)

    core_sws = []
    for i in range(4):
        sw = Switch(engine, 'core_{}'.format(i), c)
        sw.process_speed = 20
        core_sws.append(sw)
        G.add_node(sw.dp_id, instance=sw, type='core', id=i + 1)

    for pod in range(4):
        agg_sws = []
        edge_sws = []
        for i in range(2):
            sw = Switch(engine, 'agg_{}_{}'.format(pod, i), c)
            agg_sws.append(sw)
            G.add_node(sw.dp_id, instance=sw, type='agg', id=(pod + 1) * 4 + i + 1)
        for i in range(2):
            sw = Switch(engine, 'edge_{}_{}'.format(pod, i), c)
            edge_sws.append(sw)
            G.add_node(sw.dp_id, instance=sw, type='edge', id=(pod + 1) * 4 + i + 3)
            for j in range(2):
                host = Host(engine, 'host_{}_{}_{}'.format(pod, i, j), flow_list)
                G.add_node(host.addr, instance=host, type='host', id=21+pod*4+i*2+j)
                connect_node(G, sw, host)

        # 将核心交换机和聚合交换机连接
        for i in range(4):
            connect_node(G, core_sws[i], agg_sws[i // 2])

        # 将本pod内的聚合交换机和边缘交换机连接
        for i in range(2):
            for j in range(2):
                connect_node(G, agg_sws[i], edge_sws[j])

    # 将G绘制出来
    # nx.draw(G, with_labels=True)
    # plt.show()

    return G


def gen_flow(large_flow_num=5, small_flow_num=50):
    def get_pod(dp_id):
        return int(dp_id.split('_')[1])

    # host dp_id集合
    ip_list = []
    for pod in range(4):
        for i in range(2):
            for j in range(2):
                host = 'host_{}_{}_{}'.format(pod, i, j)
                ip_list.append(host)

    # 端口号集合
    src_port_list = list(range(10000, 20000))
    dst_port_list = list(range(20000, 30000))

    # 已使用的端口号集合
    used_src_ports = set()
    used_dst_ports = set()

    flow_list = []
    flows = dict()

    # 生成指定数量的流
    for i in range(large_flow_num + small_flow_num):
        # 生成源IP地址
        src_ip = random.choice(ip_list)
        src_pod = get_pod(src_ip)

        # 从未使用的端口号集合中随机选择一个源端口号
        src_port = random.choice(list(set(src_port_list) - used_src_ports))

        # 将已使用的端口号加入到已使用的端口号集合
        used_src_ports.add(src_port)

        # 生成目的IP地址
        dst_ip = random.choice(ip_list)
        dst_pod = get_pod(dst_ip)
        while src_pod == dst_pod:
            dst_ip = random.choice(ip_list)
            dst_pod = get_pod(dst_ip)

        # 从未使用的端口号集合中随机选择一个目的端口号
        dst_port = random.choice(list(set(dst_port_list) - used_dst_ports))

        # 将已使用的端口号加入到已使用的端口号集合
        used_dst_ports.add(dst_port)

        # 生成流量大小
        is_large_flow = i < large_flow_num
        if is_large_flow:
            size = random.randint(50000, 500000)
        else:
            size = random.randint(6000, 15000)

        # 计算流量开始时间和结束时间
        start_time = 50 + random.random() * 10

        # 将流信息存储到列表中
        flow_list.append({
            'start_time': start_time,
            'src_ip': src_ip,
            'src_port': src_port,
            'dst_ip': dst_ip,
            'dst_port': dst_port,
            'size': size,
            'remain_size': size
        })
        # flow_list.append((start_time, src_ip, src_port, dst_ip, dst_port, size))
        # f.write("%f\t%s\t%d\t%s\t%d\tudp\t%d\n" % (start_time, src_ip, src_port, dst_ip, dst_port, size))

        # 将流大小存储起来，以备后用
        flows[src_port] = size

    return flow_list, flows


class Simulator:
    def __init__(self):
        self.engine = Engine()
        self.engine.req_queue = queue.Queue()
        self.engine.res_queue = queue.Queue()

        self.flow_list, self.flows = gen_flow()
        self.G = build_fattree_network(self.engine, self.flow_list)

        for n in self.G.nodes:
            print(n, self.G.nodes[n]['type'], self.G.nodes[n]['id'])

    def run(self):
        self.engine.run()


def main():
    sim = Simulator()
    t = threading.Thread(target=sim.engine.run)
    t.start()

    # sim.run()

    while True:
        # if not t.is_alive():
        #     break
        try:
            req = sim.engine.req_queue.get(block=True, timeout=0.01)
        except queue.Empty:
            if not t.is_alive():
                break
            else:
                continue
        if req is not None:
            # time.sleep(2)
            sim.engine.res_queue.put(random.randint(0, 3))

    fct_list = []
    size_list = []
    pkt_num_list = []
    for flow in sim.flow_list:
        fct_list.append(flow['fct'])
        size_list.append(flow['size'])

        # 根据流的大小计算该流的包数，假设每个包的大小为1400字节，剩余的字节不足1400字节的，也算作一个包
        pkt_num = flow['size'] // 1400
        if flow['size'] % 1400 != 0:
            pkt_num += 1
        pkt_num_list.append(pkt_num)

    print('average fct per size: ', sum(fct_list) / sum(size_list) * 10000)
    print('average fct per pkt_num: ', sum(fct_list) / sum(pkt_num_list))


if __name__ == '__main__':
    main()
