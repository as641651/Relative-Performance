import numpy as np
import json
import random


class ComparisionsContainer:
    def __init__(self):
        self.data = {}

    def apply_correction(self,key1,key2,val):
        if(int(key1)>int(key2)):
            key1,key2 = key2,key1
            if val==2:
                val=0
            elif val==0:
                val=2
        return [key1,key2,val]

    def add_data(self,key1,key2,val):
        key1,key2,val = self.apply_correction(key1,key2,val)
        if not key1 in self.data:
            self.data[key1] = {}
        self.data[key1][key2] = val

    def get_data(self,key1, key2):
        k1,k2 = sorted([key1,key2])
        val = self.data[k1][k2]
        _,_,val = self.apply_correction(key1,key2,val)
        return val

    def get_data_dict(self,key):
        ret = self.data[key]
        for k,v in self.data.items():
            if key in v.keys():
                val = self.data[k][key]
                _,_,ret[k] = self.apply_correction(key,k,val)

        return ret


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def reduce_sample_size(data,size):
    out = {}
    for key in data.keys():
        out[key] = random.sample(data[key],size)
    return out


def shuffle_map(data):
    arr = list(range(len(data)))
    arr_shuffled = list(range(len(data)))
    random.shuffle(arr_shuffled)
    id_map = {}
    data_s = {}
    for i,j in enumerate(arr_shuffled):
        id_map[i] = j
        data_s[str(i)] = data[str(j)].copy()
    #print(id_map)

    return id_map, data_s
