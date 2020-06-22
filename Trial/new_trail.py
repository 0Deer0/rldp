import nni

import tensorflow as tf
import numpy as np
import json
import sys
import os
import time
import subprocess

import argparse
import logging
import math


FLAGS = None
batchsize = 1
num_layers = 2
HIDDEN_SIZE = 256
attention_mechanism = "bahdanau"
DISPLAY_REWARD_THRESHOLD = 400
RENDER = False
logger = logging.getLogger('mnist_AutoML')


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


if __name__=='__main__':

    data_path = 'graph_1_test.txt'
    data_init_path = 'graph_0_test.txt'
    op_type_group_path = 'op_group_test.txt'

    data_file = open(data_path)
    data_lines = data_file.readlines()
    op_type_group_file = open(op_type_group_path)
    op_type_groups = op_type_group_file.readlines()

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

    op_types = []
    for op in all_op_types:
        if op not in op_types:
            op_types.append(op)

    assert len(operation_names) == len(operation_ref_types)
    assert len(operation_names) == len(operation_links)

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

    # Group operations

    cnt = 0
    sum_reward = 0
    i_episode = 0
    data = 'null'
    sample_size = 1

    logger.debug('Generate data over.')
    try:
        logger.debug('report trial data')
        # weight = pipe_interface.report_trial_data()
        # #weight = [json.loads(weight)]
        # logger.debug('Receive policy from actor %s.'%weight)
        # weight = json.loads(weight)['actions']
        weight = nni.get_next_parameter()['actions']
        logger.debug('Receive policy from actor %s.'%weight)
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
        group_all_len = 0
        data = []
        logger.debug('Begin to rollout.\n')

        # get baseline time
        os.system('cd /data/data6/v-bzhang/rldp/Trial/inception && python3 test_net_baseline.py')
        baseline_run_time_reward = open('inception/running_time.txt', 'r')
        time_line = baseline_run_time_reward.readline()
        logger.debug('baseline time: %s'%time_line)
        baseline_time = float(time_line.strip('\n'))
        baseline_run_time_reward.close()
        os.system('cd /data/data6/v-bzhang/rldp/Trial/inception && rm replace*.txt running_time.txt')
        #os.system('rm inception/running_time.txt')

        init_file_lines = []
        for k in range(sample_size):
            replace_init_file = open('/data/data6/v-bzhang/rldp/Trial/inception/replace_init.txt', 'w')
            replace_train_file = open('/data/data6/v-bzhang/rldp/Trial/inception/replace_train.txt', 'w')
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
            os.system('cd /data/data6/v-bzhang/rldp/Trial/inception && python3 test_net.py')
            stop = time.time()
            run_time_reward = open('/data/data6/v-bzhang/rldp/Trial/inception/running_time.txt', 'r')
            line = run_time_reward.readline()
            logger.debug('time: %s'%line)
            run_time = float(line.strip('\n'))
            data.append([run_time])
            run_time_reward.close()
            os.system('cd /data/data6/v-bzhang/rldp/Trial/inception && rm replace*.txt running_time.txt')

        min_data = 999999
        for time in data:
            min_data = min(min_data, time[0])
           
        logger.debug('Rollout data %s.'%data)
        # print("reward: ", rewards)
        # pipe_package = generate_intermediate_package()
        # content = json.dumps({'test_acc': min_data})  # TODO
        # logger.debug('dump intermediate result ' + str(content))
        # pipe_interface.send_request_with_content(pipe_package, content)
        # logger.debug('Pipe send intermediate result done.')

        # pipe_package = generate_final_package()
        # content = package_output(data, min_data)
        # logger.debug('Final result is %s'%data)
        # pipe_interface.send_request_with_content(pipe_package, content)
        nni.report_final_result(min_data)
        logger.debug('Send final result done.')
    
    except:
        logger.exception('Got some exception in while loop in trial.py')
        raise
