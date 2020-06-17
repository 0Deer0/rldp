import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import embedding_ops
import numpy as np
import random
import json
import sys
import os
import time
import subprocess

import argparse
import logging
import math
import tempfile
import time

import nni

import command

FLAGS = None

logger = logging.getLogger('mnist_AutoML')
# r = requests.post("http://10.150.144.61:60053/debug", json={'debug_info':'import basic over.'})
#sys.path.append(os.environ['NNI_PY_SDK_DIR'])
#import pipe_interface
#import logger
#logger = logger.Logger(os.environ['NNI_LOG_PATH'])
logger.debug('Generate debug file done.')

batchsize = 1
num_layers = 2
HIDDEN_SIZE = 256
attention_mechanism = "bahdanau"
# Path to the data txt file on disk.

DISPLAY_REWARD_THRESHOLD = 400
RENDER = False

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
        '''for key in rs.keys():
            print(rs[key])'''
        return rs

def generate_intermediate_package():
    pipe_package = pipe_interface.PipePackage()
    pipe_package.command = command.CommandType.ReportTrialIntermediateVectorPath.value
    pipe_package.content_type = command.ContentType.Json.value
    return pipe_package

def generate_final_package():
    pipe_package = pipe_interface.PipePackage()
    pipe_package.command = command.CommandType.WaitTrialResult.value
    pipe_package.content_type = command.ContentType.Json.value
    return pipe_package

def package_output(data, acc):
    result = {'sample_res': data, 'output':float(acc)}
    return json.dumps(result)

if __name__=='__main__':

    data_path = 'graph_1_test.txt'
    data_init_path = 'graph_0_test.txt'
    op_type_group_path = 'op_group_test.txt'

    data_file = open(data_path)
    data_lines = data_file.readlines()
    op_type_group_file = open(op_type_group_path)
    op_type_groups = op_type_group_file.readlines()

    # print(data_lines[0])

    # Read train graph
    operation_lines = []
    operation_names = []
    operation_ids = []
    operation_links = []
    all_op_types = []
    operation_ref_types = []
    for line in data_lines:
        if line.startswith('['):
            line_operation_name = line.split('][')[2]
            line_op_type = line.split('][')[1]
            if not line_op_type == 'NoOp':
                line_operation_id = int(line.split('][')[0].split('[')[1])
                line_op_ref_type = int(line.split('{')[1].split(',')[0])
                seqs = line.split('\t')
                line_operation_link = []
                for i in range(len(seqs)):
                    if not i == 0:
                        line_operation_link.append(int(seqs[i].split('||')[0]))
                operation_lines.append(line)
                operation_ids.append(line_operation_id)
                operation_names.append(line_operation_name)
                operation_links.append(line_operation_link)
                all_op_types.append(line_op_type)
                operation_ref_types.append(line_op_ref_type)
    # print(operation_ids)
    #logger.debug('operation names num: %d'%len(operation_names))
    '''print(operation_names)
    print(operation_ref_types)'''
    # print(operation_links)

    op_types = []
    for op in all_op_types:
        if op not in op_types:
            op_types.append(op)

    assert len(operation_names) == len(operation_ref_types)
    assert len(operation_names) == len(operation_links)

    # group op
    '''op_groups = []
    for i in range(len(operation_names)):
        op_groups.append([i])

    for i in range(len(operation_ref_types)):
        if operation_ref_types[i] >= 100:
            link_indexes = []
            for link in operation_links[i]:
                for j in range(len(operation_ids)):
                    if operation_ids[j] == link:
                        link_indexes.append(j)
            for k in link_indexes:
                op_groups[i].append(k)
    u=unionfind(op_groups)
    u.createtree()
    final_groups = u.printree()

    aft_group_ops = []
    groups = []
    aft_group_ops_indexes = []

    for key in final_groups.keys():
        aft_group_ops.append(all_op_types[final_groups[key][-1]])
        groups.append(final_groups[key][-1])
        aft_group_ops_indexes.append(final_groups[key])'''
    # logger.debug(aft_group_ops)

    # read init graph
    data_init_file = open(data_init_path)
    data_init_lines = data_init_file.readlines()
    init_operation_names = []
    init_operation_ids = []
    init_operation_ref_types = []
    init_operation_link = []
    init_id_dict = dict()
    index = 0
    for init_line in data_init_lines:
        if init_line.startswith('['):
            init_line_operation_name = init_line.split('][')[2]
            init_line_op_type = init_line.split('][')[1]
            if not init_line_op_type == 'NoOp':
                init_line_operation_id = int(init_line.split('][')[0].split('[')[1])
                init_line_op_ref_type = int(init_line.split('{')[1].split(',')[0])
                init_seqs = init_line.split('\t')
                init_line_operation_link = []
                for i in range(len(init_seqs)):
                    if not i == 0:
                        init_line_operation_link.append(int(init_seqs[i].split('||')[0]))
                init_id_dict[init_line_operation_id] = index
                init_operation_ids.append(init_line_operation_id)
                init_operation_names.append(init_line_operation_name)
                init_operation_ref_types.append(init_line_op_ref_type)
                init_operation_link.append(init_line_operation_link)
                index += 1

    #logger.debug('op type groups num: %d'%len(op_type_groups))

    # read group op
    feed_ops = []
    tmp_operation_tokens = [False]*len(operation_names)
    replace_op_groups = []
    for j in range(len(op_type_groups)):
        replace_op_groups.append([])
        op_prefix = op_type_groups[j].strip('\n').split(' ')[0]
        feed_ops.append(op_prefix)
        for i in range(len(operation_names)):
            if tmp_operation_tokens[i] == False and operation_names[i] not in replace_op_groups[j] and operation_names[i].startswith(op_prefix) and not operation_names[i].startswith(op_prefix + '_') and not operation_names[i].startswith(op_prefix + '0') and not operation_names[i].startswith(op_prefix + '1') and not operation_names[i].startswith(op_prefix + '2') and not operation_names[i].startswith(op_prefix + '3') and not operation_names[i].startswith(op_prefix + '4') and not operation_names[i].startswith(op_prefix + '5') and not operation_names[i].startswith(op_prefix + '6') and not operation_names[i].startswith(op_prefix + '7') and not operation_names[i].startswith(op_prefix + '8') and not operation_names[i].startswith(op_prefix + '9'):
                replace_op_groups[j].append(operation_names[i])
                tmp_operation_tokens[i] = True
        logger.debug('j: %d'%j)
        logger.debug('replace op groups[j]: %s'%replace_op_groups[j])

    '''op_num = len(aft_group_ops)
    op_type_num = len(op_types)
    op_output_shapes = []
    max_output_shape_len = 3
    op_adjs = []

    # fake output shapes
    for i in operation_ids:
        op_output_shapes.append([int(random.random() * 10), int(random.random() * 10), int(random.random() * 10)])

    # real adjs
    for link in operation_links:
        op_adj = np.zeros_like(operation_ids)
        for op in link:
            for i in range(len(operation_ids)):
                if operation_ids[i] == op:
                    op_adj[i] = 1
        op_adjs.append(op_adj)

    # print(op_types)
    # print(op_adjs)
    type_emb_size = 64

    devices = ['cpu-0', 'cpu-1', 'gpu-0', 'gpu-1']
    devices_types = ['cpu-0', 'cpu-1', 'gpu-0', 'gpu-1']
    device_num = 4
    device_type_num = 4
    device_emb_size = 64
    # decoder_size = op_num + 1 # including start and end tag


    type_emb = np.zeros(
        (1, op_num),
        dtype='float32')
    output_shapes_arr = np.zeros(
        (1, op_num, max_output_shape_len),
        dtype='float32')
    adjs = np.zeros(
        (1, op_num, op_num),
        dtype='float32')
    device_emb = np.zeros(
        (1, device_num),
        dtype='float32')

    for i in range(op_num):
        for j in range(op_type_num):
            if aft_group_ops[i] == op_types[j]:
                type_emb[0, i] = j

    for i in range(op_num):
        output_shapes_arr[0, i] = op_output_shapes[i]

    for i in range(op_num):
        adjs[0, i] = op_adjs[i]

    # start token
    tgt_sos_id = [[0]*type_emb_size]'''

    # Group operations

    cnt = 0
    sum_reward = 0
    i_episode = 0
    data = 'null'
    sample_size = 1

    logger.debug('Generate data over.')
    try:
        logger.debug('report trial data')
        weight = pipe_interface.report_trial_data()
        #weight = [json.loads(weight)]
        logger.debug('Receive policy from actor %s.'%weight)
        weight = json.loads(weight)['actions']
        dropout_rate = 0.5
        lr = 0.001
        actions = []
        
        for i in range(sample_size):
            actions.append([])
            for action_arr in weight[i][0]:
                for index in range(len(action_arr)):
                    if action_arr[index] == 1:
                        actions[i].append(index)
        
        #assert len(weight[0]) == len(actions)
        logger.debug('Actions len: %d.'%len(actions))
        logger.debug('Groups len: %d.'%len(replace_op_groups))
        #assert len(actions) == len(replace_op_groups)
        logger.debug('Actions: %s.'%actions)
        #actions = [[0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]]
        group_all_len = 0
        data = []
        logger.debug('Begin to rollout.\n')

        # get baseline time
        os.system('cd /home/v-yali6/workspace_yal/rldp/Trial/inception && python3 test_net_baseline.py')
        baseline_run_time_reward = open('inception/running_time.txt', 'r')
        time_line = baseline_run_time_reward.readline()
        logger.debug('baseline time: %s'%time_line)
        baseline_time = float(time_line.strip('\n'))
        baseline_run_time_reward.close()
        os.system('cd /home/v-yali6/workspace_yal/rldp/Trial/inception && rm replace*.txt running_time.txt')
        #os.system('rm inception/running_time.txt')

        init_file_lines = []
        for k in range(sample_size):
            replace_init_file = open('/home/v-yali6/workspace_yal/rldp/Trial/inception/replace_init.txt', 'w')
            replace_train_file = open('/home/v-yali6/workspace_yal/rldp/Trial/inception/replace_train.txt', 'w')
            for i in range(len(replace_op_groups)):
                group_all_len += len(replace_op_groups[i])
                for j in range(len(replace_op_groups[i])):
                    replace_train_file.write(replace_op_groups[i][j] + ' ' + str(actions[k][i]) + '\n')
                    for init_op_index in range(len(init_operation_names)):
                        #print(init_operation_names[init_op_index])
                        #print(replace_op_groups[i][j])
                        if init_operation_names[init_op_index] == replace_op_groups[i][j]:
                            #print(init_operation_names[init_op_index])
                            #init_file_lines.append(init_operation_names[init_op_index])
                            replace_init_file.write(init_operation_names[init_op_index] + ' ' + str(actions[k][i]) + '\n')
                            if init_operation_ref_types[init_op_index] >= 100:
                                #print(init_operation_names[init_op_index])
                                #print(init_operation_ref_types[init_op_index])
                                for id in init_operation_link[init_op_index]:
                                    #print('link ids: ')
                                    #print(init_operation_names[id-2])
                                    #if init_operation_names[id-2] not in init_file_lines:
                                    replace_init_file.write(init_operation_names[init_id_dict[id]] + ' ' + str(actions[k][i]) + '\n')
                                #print('link over.')
                            break
            replace_init_file.close()
            replace_train_file.close()

            start = time.time()
            os.system('cd /home/v-yali6/workspace_yal/rldp/Trial/inception && python3 test_net.py')
            stop = time.time()
            run_time_reward = open('/home/v-yali6/workspace_yal/rldp/Trial/inception/running_time.txt', 'r')
            line = run_time_reward.readline()
            logger.debug('time: %s'%line)
            run_time = float(line.strip('\n'))
            data.append([run_time])
            run_time_reward.close()
            os.system('cd /home/v-yali6/workspace_yal/rldp/Trial/inception && rm replace*.txt running_time.txt')

        min_data = 999999
        for time in data:
            min_data = min(min_data, time[0])
           
        logger.debug('Rollout data %s.'%data)
        # print("reward: ", rewards)
        pipe_package = generate_intermediate_package()
        content = json.dumps({'test_acc': min_data})
        logger.debug('dump intermediate result ' + str(content))
        pipe_interface.send_request_with_content(pipe_package, content)
        logger.debug('Pipe send intermediate result done.')

        pipe_package = generate_final_package()
        content = package_output(data, min_data)
        logger.debug('Final result is %s'%data)
        pipe_interface.send_request_with_content(pipe_package, content)
        logger.debug('Send final result done.')

        #os.system('rm inception/replace_train.txt && rm inception/replace_init.txt && rm inception/running_time.txt')

        # RL = DevicePlacement(op_num, op_type_num, device_num, device_type_num, type_emb_size, device_emb_size, max_output_shape_len, sample_size, dropout=dropout_rate, learning_rate=lr)

        # actions = RL.choose_action(type_emb, output_shapes_arr, adjs, tgt_sos_id, sample_size)
        # print(str(i_episode) + " episode action: ", weight)
        # run sample_size times
        
    except:
        logger.exception('Got some exception in while loop in trial.py')
        raise