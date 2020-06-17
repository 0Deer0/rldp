# -*- encoding:utf8 -*-

from __future__ import print_function

import json
import sys

def parse_trial_results(data, input_name = 'input', output_name = 'output'):
    x = []
    y = []
    z = []
    for temp in data:
        #if temp is None:
        if temp == 'null':
            continue
        x.append(temp[input_name])
        y.append(temp[output_name]['output'])
        z.append(temp[output_name]['sample_res'])
    return x, y, z

def parse_request(data):
    # TODO: xuehui,qiyu, please change this assert line
    #assert data['count']
    count = data['count']
    x = None
    y = None
    z = None

    if 'array' in data:
        array = data['array']
        x, y, z = parse_trial_results(array)
    return count, x, y, z

def parse_params_request(data):
    return int(data)

def package_parameters(paramters):
    result = json.dumps(paramters)
    return result