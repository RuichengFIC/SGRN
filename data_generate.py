import numpy as np
from utils import *
import itertools

def generate_patterns(sample_num, watch_length, pattern_num, patterns,distort_level = 0.5):
# task 1
    x = []
    pattern_positions = [] 
    pattern_info = []
    for _ in range(sample_num): 
        pattern_info.append(np.random.choice(range(len(patterns)),pattern_num,replace = False))
    pattern_info = np.array(pattern_info)
    for i in range(sample_num):
        pattern_specs = [distort_seq(x, distort_level) for x in patterns]
        left_space = watch_length
        for j in range(pattern_num):
            left_space = left_space - len(pattern_specs[pattern_info[i][j]])                                 
        seq = np.array([])
        position =  [-1] * pattern_num
        cur = 0
        for j in range(pattern_num):  # 根据顺序
            start_num = np.random.randint(0,left_space + 1)
            seq = np.concatenate([seq,(np.random.rand(start_num) - 0.5) * 2])
            cur = cur + start_num
            left_space = left_space - start_num
            position[j] = cur
            seq = np.concatenate([seq, pattern_specs[pattern_info[i][j]]])
            cur = cur + len(pattern_specs[pattern_info[i][j]])
        seq = np.concatenate([seq,(np.random.rand(left_space) - 0.5) * 2])  
        pattern_positions.append(position)
        x.append(seq)
    return np.array(x), np.array(pattern_positions)

def generate_patterns_1(sample_num, watch_length, patterns,distort_level = 0.5):
    x_1 = []
    y_1 = []
    x_1.append((np.random.rand(sample_num,watch_length) - 0.5) * 2)
    y_1.append([0] * sample_num)
    for pattern_num in range(1,len(patterns)+1):
        x_, pos_ = generate_patterns(sample_num = sample_num, watch_length = watch_length, pattern_num = pattern_num,distort_level = 0.5, patterns = patterns)
        x_1.append(x_)
        y_1.append([pattern_num]* sample_num)
    x_1 = np.concatenate(x_1, axis = 0)
    y_1 = np.concatenate(y_1, axis = 0)
    return x_1, y_1

def generate_patterns_2(sample_num, watch_length, patterns,distort_level = 0.5):
# task 2
    x = []
    y = []
    # 27
    for n, pattern_order in enumerate(itertools.product(range(3),repeat = 3)):
#         print(pattern_order)
        pattern_info = []
        # 300
        for _ in range(sample_num): 
            pattern_info.append(np.random.choice(range(2),3,replace = True))
        pattern_info = np.array(pattern_info)
        for i in range(sample_num):
            left_space = watch_length
            pattern_specs = []
            for j in range(len(pattern_order)):
                cur_pattern = distort_seq(patterns[pattern_order[j] * 2 + pattern_info[i,j]],distort_level)
                pattern_specs.append(cur_pattern)
                left_space = left_space - len(cur_pattern)            
                                          
            seq = np.array([])
            position =  [-1] * 3
            cur = 0
            for p in pattern_specs:  # 根据顺序
                start_num = np.random.randint(0,left_space + 1)
                seq = np.concatenate([seq,(np.random.rand(start_num) - 0.5) * 2])
                cur = cur + start_num
                left_space = left_space - start_num
                position[j] = cur
                seq = np.concatenate([seq, p])
                cur = cur + len(p)
            seq = np.concatenate([seq,(np.random.rand(left_space) - 0.5) * 2])  
            x.append(seq)
        y = y + [n] * sample_num
    return np.array(x), np.array(y)

def generate_patterns_3(sample_num, watch_length, patterns,distort_level = 0.5):
# task 3
    x = []
    y = []
    # 27
    for n, lag_intervel in enumerate(list(range(5,31,5))):
        for _ in range(sample_num):
            selected_patterns = np.random.choice(range(6),2,replace = True)
            left_space = watch_length
            pattern_specs = []
            for p in selected_patterns:
                cur_pattern = distort_seq(patterns[p],distort_level)
                pattern_specs.append(cur_pattern)
                left_space = left_space - len(cur_pattern)            
            left_space = left_space - lag_intervel
            
            seq = np.array([])
            cur = 0
        
            start_num = np.random.randint(0,left_space + 1)
            seq = np.concatenate([seq,(np.random.rand(start_num) - 0.5) * 2])   
            left_space = left_space - start_num
            
            seq = np.concatenate([seq, pattern_specs[0]])
            
            seq = np.concatenate([seq,(np.random.rand(lag_intervel) - 0.5) * 2])

            seq = np.concatenate([seq, pattern_specs[1]])
            
            seq = np.concatenate([seq,(np.random.rand(left_space) - 0.5) * 2])  
            x.append(seq)
        y = y + [n] * sample_num
    return np.array(x), np.array(y)