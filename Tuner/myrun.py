import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import embedding_ops
import numpy as np
import random
import json
import json_parser
import sys
import os
import time
from data_process import Operation, unionfind

batchsize = 1
num_layers = 4
HIDDEN_SIZE = 64
attention_mechanism = "bahdanau"
# Path to the data txt file on disk.

DISPLAY_REWARD_THRESHOLD = 400
RENDER = False

class DevicePlacement(object):

    def __init__(self, op_num, op_type_num, device_num, device_type_num, type_emb_size, device_emb_size, max_output_shape_len, sample_size, dropout=0.5, learning_rate=0.001, reward_decay=0.9):
        self.op_num = op_num
        self.device_num = device_num
        self.op_type_num = op_type_num
        self.device_type_num = device_type_num
        self.type_emb_size = type_emb_size
        self.device_emb_size = device_emb_size
        self.sample_size = sample_size
        self.max_output_shape_len = max_output_shape_len

        #self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self.lr = learning_rate
        self.gamma = reward_decay
        self.dropout = dropout

        self.define_device_placement_network()

        self.sess = tf.Session()
        
        self.writer = tf.summary.FileWriter('/tmp/my_rl_graph', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())


    def define_device_placement_network(self):
        with tf.variable_scope("device_placement",reuse=tf.AUTO_REUSE):
            self.input_data = tf.placeholder(dtype=tf.int32, shape=[batchsize, self.op_num])
            input_embedding = tf.get_variable("input_embedding", [self.op_type_num, self.type_emb_size])
            self.type = tf.nn.embedding_lookup(input_embedding, self.input_data)
            self.output_shape = tf.placeholder(dtype=tf.float32, shape=[batchsize, self.op_num, self.max_output_shape_len])
            self.adj = tf.placeholder(dtype=tf.float32, shape=[batchsize, self.op_num, self.op_num])
            self.observations = tf.concat([tf.concat([self.type, self.output_shape], 2), self.adj], 2)
            self.value = tf.placeholder(dtype=tf.float32, shape=[self.sample_size, batchsize, ], name="actions_value")
            self.output_layer = layers_core.Dense(self.device_num) # projection layer

            def lstm(inputs, hidden_size, num_layers, dropout, name, initial_state=None):
                def dropout_lstm_cell():
                    return tf.contrib.rnn.DropoutWrapper(
                        tf.contrib.rnn.BasicLSTMCell(hidden_size),
                        input_keep_prob=dropout)
                layers = [dropout_lstm_cell() for _ in range(num_layers)]
                with tf.variable_scope(name):
                    return tf.nn.dynamic_rnn(
                        tf.contrib.rnn.MultiRNNCell(layers),
                        inputs,
                        initial_state=initial_state,
                        dtype=tf.float32,
                        time_major=False)

            def lstm_bid_encoder(inputs, hidden_size, num_hidden_layers, dropout, name):
                """Bidirectional LSTM for encoding inputs that are [batch x time x size]."""

                def dropout_lstm_cell():
                    return tf.contrib.rnn.DropoutWrapper(
                        tf.contrib.rnn.BasicLSTMCell(hidden_size),
                        input_keep_prob=dropout)

                with tf.variable_scope(name):
                    cell_fw = tf.contrib.rnn.MultiRNNCell(
                        [dropout_lstm_cell() for _ in range(num_hidden_layers)])

                    cell_bw = tf.contrib.rnn.MultiRNNCell(
                        [dropout_lstm_cell() for _ in range(num_hidden_layers)])

                    ((encoder_fw_outputs, encoder_bw_outputs),
                    (encoder_fw_state, encoder_bw_state)) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=cell_fw,
                        cell_bw=cell_bw,
                        inputs=inputs,
                        dtype=tf.float32,
                        time_major=False)

                    encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
                    encoder_states = []

                    for i in range(num_hidden_layers):
                        if isinstance(encoder_fw_state[i], tf.contrib.rnn.LSTMStateTuple):
                            encoder_state_c = tf.concat(
                                values=(encoder_fw_state[i].c, encoder_bw_state[i].c),
                                axis=1,
                                name="encoder_fw_state_c")
                            encoder_state_h = tf.concat(
                                values=(encoder_fw_state[i].h, encoder_bw_state[i].h),
                                axis=1,
                                name="encoder_fw_state_h")
                            encoder_state = tf.contrib.rnn.LSTMStateTuple(
                                c=encoder_state_c, h=encoder_state_h)
                        elif isinstance(encoder_fw_state[i], tf.Tensor):
                            encoder_state = tf.concat(
                                values=(encoder_fw_state[i], encoder_bw_state[i]),
                                axis=1,
                                name="bidirectional_concat")

                        encoder_states.append(encoder_state)

                    encoder_states = tuple(encoder_states)
                    return encoder_outputs, encoder_states

            def lstm_attention_decoder(embedding_decoder, hidden_size, dropout, num_layers, num_heads, attention_mechanism, 
                                    attention_layer_size, output_attention, name, initial_state, 
                                    encoder_outputs, start_token, projection_layer, maximum_iterations):
                """Run LSTM cell with attention on inputs of shape [batch x time x size]."""

                def dropout_lstm_cell():
                    return tf.contrib.rnn.DropoutWrapper(
                        tf.nn.rnn_cell.BasicLSTMCell(hidden_size),
                        input_keep_prob=dropout)

                layers = [dropout_lstm_cell() for _ in range(num_layers)]
                if attention_mechanism == "luong":
                    attention_mechanism_class = tf.contrib.seq2seq.LuongAttention
                elif attention_mechanism == "bahdanau":
                    attention_mechanism_class = tf.contrib.seq2seq.BahdanauAttention
                else:
                    raise ValueError("Unknown attention_mechanism = %s\n, must be "
                                "luong or bahdanau." % attention_mechanism)
                used_attention_mechanism = attention_mechanism_class(
                    hidden_size, encoder_outputs)

                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    tf.nn.rnn_cell.MultiRNNCell(layers),
                    [used_attention_mechanism]*num_heads,
                    attention_layer_size=[attention_layer_size]*num_heads,
                    output_attention=(output_attention == 1))


                # Helper
                '''helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding_decoder,
                    tf.fill([batchsize], tgt_sos_id), tgt_eos_id)'''

                decoder_initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=batchsize).clone(cell_state=initial_state)
                
                if callable(embedding_decoder):
                    embedding_fn = embedding_decoder
                else:
                    embedding_fn = (
                        lambda ids: embedding_ops.embedding_lookup(embedding_decoder, ids))

                outputs = []
                input_tensor = start_token
                for i in range(maximum_iterations):
                    cell_output, state = decoder_cell(input_tensor, decoder_initial_state)
                    cell_output = projection_layer(cell_output)
                    sample_ids = tf.argmax(cell_output, axis=-1, output_type=tf.int32)
                    seqoutput = tf.expand_dims(cell_output, 1)
                    outputs.append(seqoutput)
                    input_tensor = embedding_fn(sample_ids)
                    decoder_initial_state = state
                
                rnn_outputs = tf.concat(outputs, 1)
                print(rnn_outputs)

                # Decoder
                '''decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, helper, initial_state,
                    output_layer=projection_layer)
                # Dynamic decoding
                with tf.variable_scope(name):
                    outputs, state, _ = tf.contrib.seq2seq.dynamic_decode(
                        decoder, maximum_iterations=maximum_iterations)'''

                #for i in range(maximum_iterations):
                #    step_output, step_state = decoder.step()

                # For multi-head attention project output back to hidden size
                # if output_attention == 1 and num_heads > 1:
                #     outputs = tf.layers.dense(outputs.rnn_output, hidden_size)

                return rnn_outputs, state

            self.start_token = tf.placeholder(dtype=tf.float32, shape=[batchsize, self.device_emb_size])
            decoder_input_embedding = tf.get_variable("decoder_input_embedding", [self.device_type_num, self.device_emb_size])
            # self.decoder_input = tf.nn.embedding_lookup(decoder_input_embedding, self.decoder_input_data)
            # Look up embedding:
            #   decoder_inputs: [batch_size, max_decoder_output_len]
            #   decoder_emb_inp: [batch_size, max_decoder_output_len, embedding_size]
            # actions
            self.decoder_target_data = tf.placeholder(tf.float32, [self.sample_size, batchsize, self.op_num, self.device_num])
            encoder_outputs, final_encoder_state = lstm(self.observations, HIDDEN_SIZE, num_layers, self.dropout, "encoder")
            # print(encoder_outputs, final_encoder_state)
            decoder_outputs, _ = lstm_attention_decoder(decoder_input_embedding, HIDDEN_SIZE, self.dropout, num_layers, 1, attention_mechanism, 
                HIDDEN_SIZE, True, "decoder", final_encoder_state, encoder_outputs, self.start_token, self.output_layer, self.op_num)

            #print(decoder_outputs.get_shape())
            
            # decoder_outputs = tf.reshape(decoder_outputs, [batchsize, self.device_num, self.device_emb_size])
            # print(decoder_outputs.get_shape())

            #decoder_outputs = tf.layers.dense(decoder_outputs, self.device_emb_size)
            self.all_act_prob = tf.nn.softmax(logits=decoder_outputs)
            #print(decoder_outputs.get_shape())
            #print(self.decoder_target_data.get_shape())
            self.loss = 0
            for i in range(self.sample_size):
                #pos_log_prob = tf.reduce_sum(tf.log(self.all_act_prob)*self.decoder_target_data[i], axis=-1)
                #print(pos_log_prob.get_shape())
                neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=decoder_outputs, labels=self.decoder_target_data[i])
                apply_reward = tf.multiply(self.value[i], neg_log_prob)
                self.loss += tf.reduce_mean(apply_reward)

            self.loss /= self.sample_size

            tf.summary.scalar("loss", self.loss)
            self.merged_summary = tf.summary.merge_all()
            
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = optimizer.minimize(self.loss)
        
            #self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(decoder_outputs, 2), tf.argmax(self.decoder_target_data, 2)), "float"))
    
    def choose_action(self, type_emb, output_shape, adj, tgt_sos_id, sample_size, device_type_num, random_rate=0):

        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.input_data: type_emb, self.output_shape: output_shape, self.adj: adj, self.start_token: tgt_sos_id})
        actions = []
        for k in range(sample_size):
            action = []
            for i in range(prob_weights.shape[1]):
                for t in range(len(prob_weights[0, i])):
                    prob_weights[0, i, t] = np.true_divide(prob_weights[0, i, t], 1.000001)
                #mylog.write(sum)
                #mylog.write(new_sum)
                if random.random() >= random_rate:
                    '''choose_action = []
                    for prob in prob_weights[0, i]:
                        if random.random() < prob:
                            choose_action.append(1)
                        else:
                            choose_action.append(0)
                    action.append(choose_action)'''
                    action.append(np.random.multinomial(1, prob_weights[0, i, :]).tolist())
                else:
                    tmp_random = [0]*device_type_num
                    tmp_random[int(random.random()*10) % device_type_num] = 1
                    action.append(tmp_random)
            actions.append([action])
        # print(actions)
        return actions

    '''def store_transition(self, o, a, r):
        self.ep_obs.append(o)
        self.ep_as.append(a)
        self.ep_rs.append(r)'''

    def learn(self, type_emb, output_shape, adj, tgt_sos_id, actions, rewards):

        # train on episode
        _, all_act_prob, summary, loss = self.sess.run([self.train_op, self.all_act_prob, self.merged_summary, self.loss], feed_dict={
             self.input_data: type_emb, 
             self.output_shape: output_shape, 
             self.adj: adj,
             self.start_token: tgt_sos_id,
             self.decoder_target_data: actions,  # shape=[None, ]
             self.value: rewards  # shape=[None, ]
        })

        #self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return all_act_prob, summary, loss
    '''def _discount_and_norm_rewards(self, rewards):
        discounted_ep_rs = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_ep_rs[t] = running_add

        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs'''

if __name__=='__main__':

    data_path = 'graph_1.txt'
    data_init_path = 'graph_0.txt'
    op_type_group_path = 'op_group.txt'

    data_file = open(data_path)
    data_lines = data_file.readlines()

    op_type_group_file = open(op_type_group_path)
    op_type_groups = op_type_group_file.readlines()
    # print(data_lines[0])
    
    # pre process
    operation_lines = []
    operation_names = []
    operation_ids = []
    operation_links = []
    all_op_types = []
    operation_ref_types = []

    operation_objs = []

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
    # mylog.write(operation_names)
    # mylog.write(operation_ref_types)
    # print(operation_links)

    op_types = []
    for op in all_op_types:
        if op not in op_types:
            op_types.append(op)

    assert len(operation_names) == len(operation_ref_types)
    assert len(operation_names) == len(operation_links)

    #print(len(operation_names))
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

    # group op
    op_groups = dict()
    for line in op_type_groups:
        line = line.strip('\n').split(' ')[0]
        op_groups[line] = []
    for i in range(len(operation_names)):
        for line in op_type_groups:
            line = line.strip('\n').split(' ')[0]
            if operation_names[i].startswith(line) and not operation_names[i].startswith(line + '_'):
                tmp_op = Operation(i, operation_names[i], all_op_types[i], line, operation_links[i])
                op_groups[line].append(tmp_op)
                break

    tmp_operation_tokens = [False]*len(operation_names)
    replace_op_groups = []
    for j in range(len(op_type_groups)):
        replace_op_groups.append([])
        op_prefix = op_type_groups[j].strip('\n').split(' ')[0]
        for i in range(len(operation_names)):
            if tmp_operation_tokens[i] == False and operation_names[i] not in replace_op_groups[j] and operation_names[i].startswith(op_prefix) and not operation_names[i].startswith(op_prefix + '_') and not operation_names[i].startswith(op_prefix + '0') and not operation_names[i].startswith(op_prefix + '1') and not operation_names[i].startswith(op_prefix + '2') and not operation_names[i].startswith(op_prefix + '3') and not operation_names[i].startswith(op_prefix + '4') and not operation_names[i].startswith(op_prefix + '5') and not operation_names[i].startswith(op_prefix + '6') and not operation_names[i].startswith(op_prefix + '7') and not operation_names[i].startswith(op_prefix + '8') and not operation_names[i].startswith(op_prefix + '9'):
                replace_op_groups[j].append(operation_names[i])
                tmp_operation_tokens[i] = True   

    # print(op_groups)
    timer = 0
    for key, val in op_groups.items():
        #print('group is ' + key)
        for obj in val:
            assert obj.group == key
            infoline = ''
            infoline += '[' + str(timer) + '][' + obj.name + '][' + obj.type + '][' + obj.group + ']['
            for i in range(len(obj.next)):
                infoline += str(obj.next[i]) + ','
            infoline += '][' + str(obj.output_shape) + ']'
            timer += 1
            #print(infoline)

    feed_ops = []
    feed_all_op_types = []
    feed_op_output_shapes = []
    feed_op_links = []
    feed_op_ids=[]
    for line in op_type_groups:
        line = line.strip('\n').split(' ')
        feed_op_name = line[0]
        feed_op_type = line[-1]
        output_links = []
        for link in line[2].split(','):
            if link == 'null':
                output_links.append(-1)
            else:
                output_links.append(int(link))
        '''for inlink in line[1].split(','):
            if inlink == 'null':
                output_links.append(-1)
            else:
                output_links.append(int(inlink))'''
            #print(output_links)
        shape = []
        for dim in line[3].split(','):
            shape.append(int(dim))
            #print(shape)
        feed_ops.append(feed_op_name)
        feed_all_op_types.append(feed_op_type)
        feed_op_links.append(output_links)
        feed_op_output_shapes.append(shape)
    print(len(feed_ops))
        
    feed_op_types = []
    for op in feed_all_op_types:
        if op not in feed_op_types:
            feed_op_types.append(op)

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
    aft_group_ops_links = [[]]*len(final_groups.keys())
    
    timer = 0
    for key in final_groups.keys():
        aft_group_ops.append(all_op_types[final_groups[key][-1]])
        for op in final_groups[key]:
            op_link = operation_links[op]
            for one_op_link in op_link:
                for index in final_groups.keys():
                    for linked_op in final_groups[index]:
                        if index != key and one_op_link == linked_op:
                            aft_group_ops_links[timer].append(index)
        timer += 1'''
    
    #print(aft_group_ops)
    #print(aft_group_ops_links)

    #mylog.write(aft_group_ops)


    op_num = len(feed_ops)
    op_type_num = len(feed_op_types)
    op_output_shapes = feed_op_output_shapes
    max_output_shape_len = 4
    op_adjs = []

    '''op_num = len(aft_group_ops)
    op_type_num = len(op_types)
    op_output_shapes = []
    max_output_shape_len = 3
    op_adjs = []'''

    # fake output shapes
    #for i in operation_ids:
    #    op_output_shapes.append([int(random.random() * 10), int(random.random() * 10), int(random.random() * 10)])

    # real adjs
    for link in feed_op_links:
        op_adj = np.zeros(len(feed_ops))
        for op in link:
            if op != -1:
                op_adj[op] = 1
        op_adjs.append(op_adj)

    '''for link in aft_group_ops_links:
        op_adj = [0]*len(aft_group_ops)
        for op in link:
            op_adj[op] = 1
        op_adjs.append(op_adj)'''

    # print(op_types)
    # print(op_adjs)
    type_emb_size = 128

    devices = ['gpu-0', 'gpu-1', 'gpu-2']
    devices_types = ['gpu-0', 'gpu-1', 'gpu-2']
    device_num = 3
    device_type_num = 3
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
            if feed_all_op_types[i] == feed_op_types[j]:
                type_emb[0, i] = j

    for i in range(op_num):
        output_shapes_arr[0, i] = op_output_shapes[i]

    for i in range(op_num):
        adjs[0, i] = op_adjs[i]

    '''type_emb = np.zeros(
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
        adjs[0, i] = op_adjs[i]'''

    # start token
    tgt_sos_id = [[0]*device_emb_size]

    mylog = open('/home/v-yali6/workspace_yal/rldp/Tuner/rl_log.txt', 'w')

    mylog.write('Generate data over.\n')
    
    mylog.write('type: %s\n'%type_emb)
    mylog.write('outputshape: %s\n'%output_shapes_arr)
    mylog.write('adjs: %s\n'%adjs)

    cnt = 0
    i_episode = 0
    data = 'null'
    dropout_rate = 0.5
    lr = 0.0001
    sample_size = 32

    try:
        RL = DevicePlacement(op_num, op_type_num, device_num, device_type_num, type_emb_size, device_emb_size, max_output_shape_len, sample_size, dropout=dropout_rate, learning_rate=lr)
    except:
        mylog.write('Got some exception in while loop in RL obj.')
        raise
    mylog.write('Build network over.')

    episode = 1
    random_rate = 0.2
    all_weights = []
    all_rewards = []
    while True:
        try:
            if episode % 50 == 0:
                random_rate /= 2
            weight = RL.choose_action(type_emb, output_shapes_arr, adjs, tgt_sos_id, sample_size, device_type_num, random_rate=random_rate)
            actions = []
            for i in range(sample_size):
                actions.append([])
                for action_arr in weight[i][0]:
                    for index in range(len(action_arr)):
                        if action_arr[index] == 1:
                            actions[i].append(index)
            mylog.write('actions: %s\n'%actions)
        
            #assert len(weight[0]) == len(actions)
            #actions = [[0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]]
            group_all_len = 0
            data = []

            # get baseline time
            os.system('cd /home/v-yali6/workspace_yal/rldp/Trial/inception && python3 inception_baseline.py')
            baseline_run_time_reward = open('/home/v-yali6/workspace_yal/rldp/Trial/inception/running_time.txt', 'r')
            time_line = baseline_run_time_reward.readline()
            mylog.write('baseline time: %s\n'%time_line)
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
                os.system('cd /home/v-yali6/workspace_yal/rldp/Trial/inception && python3 inception_train_1.py')
                stop = time.time()
                run_time_reward = open('/home/v-yali6/workspace_yal/rldp/Trial/inception/running_time.txt', 'r')
                line = run_time_reward.readline()
                mylog.write('time: %s\n'%line)
                run_time = float(line.strip('\n'))
                data.append([baseline_time - run_time])
                run_time_reward.close()
                os.system('cd /home/v-yali6/workspace_yal/rldp/Trial/inception && rm replace*.txt running_time.txt')

            min_data = 999999
            for tmp_time in data:
                min_data = min(min_data, tmp_time[0])
            # rewards = json.loads(rewards)
            rewards = data
            #mylog.write('rewards: %s\n'%rewards)
            mylog.write('rewards: %s\n'%rewards)
            for one_act in weight:
                all_weights.append(one_act)
            for one_reward in rewards:
                all_rewards.append(one_reward)
            max_reward = -99999
            min_reward = 99999
            max_index = -1
            min_index = -2
            for i in range(len(all_rewards)):
                if all_rewards[i] > max_reward:
                    max_reward = all_rewards[i]
                    max_index = i
                if all_rewards[i] < min_reward:
                    min_reward = all_rewards[i]
                    min_index = i
            mylog.write('current best action: %s\n'%all_weights[max_index])
            mylog.write('current best reward: %s\n'%all_rewards[max_index])
            '''feed_weights = []
            feed_rewards = []
            for i in range(sample_size - 1):
                feed_weights.append(all_weights[max_index])
                feed_rewards.append(all_rewards[max_index])
            feed_weights.append(all_weights[min_index])
            feed_rewards.append(all_rewards[min_index])
            for i in range(min(100, episode*2)):
                all_act_prob, summary, loss = RL.learn(type_emb, output_shapes_arr, adjs, tgt_sos_id, feed_weights, feed_rewards)
                RL.writer.add_summary(summary, episode-1)
                mylog.write('all_act_prob: %s\n'%all_act_prob)
                #mylog.write('neg_log_prob: %s\n'%neg_log_prob)
                mylog.write('loss: %s\n'%loss)'''
            for i in range(len(all_weights) / sample_size):
                all_act_prob, summary, loss = RL.learn(type_emb, output_shapes_arr, adjs, tgt_sos_id, all_weights[i*sample_size:(i+1)*sample_size], all_rewards[i*sample_size:(i+1)*sample_size])
                RL.writer.add_summary(summary, episode-1)
                mylog.write('all_act_prob: %s\n'%all_act_prob)
                #mylog.write('neg_log_prob: %s\n'%neg_log_prob)
                mylog.write('loss: %s\n'%loss)
            #mylog.write('all_act_prob: %s\n'%all_act_prob)
            #mylog.write('neg_log_prob: %s\n'%neg_log_prob)
            #mylog.write('loss: %s\n'%loss)
            ##random_rate -= 0.02
                
        except:
            raise
        
        mylog.flush()
        episode += 1
