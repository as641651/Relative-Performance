import random
import numpy as np
import json
import sys
import operator
import argparse
from utils import ComparisionsContainer, NpEncoder, reduce_sample_size, shuffle_map


def compare(a,b,stat,thresh,m,r):
    c=0

    for i in range(r):
        ma = random.sample(a,m)
        mb = random.sample(b,m)
        if(stat(mb)<stat(ma)):
            c = c+1
    p = float(c)/float(r)

    ret = 1
    if(p>=thresh):
        ret = 2
    elif(p<(1-thresh)):
        ret = 0

    return ret


def bubbleSort(data_r, thresh,m,r):

    id_map, data = shuffle_map(data_r)

    arr = np.array(list(range(len(data))))
    n = len(arr)
    rank_arr = np.array(list(range(1,n+1)))

    container = ComparisionsContainer()
    # Traverse through all array elements
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n-i-1):
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            a_key = [str(arr[j]).zfill(3),str(arr[j+1]).zfill(3)]
            try:
                # check if the result of comparison already exists
                ret = container.get_data(a_key[0],a_key[1])
            except KeyError:
                ret = compare(data[str(arr[j])], data[str(arr[j+1])],np.min,thresh,m,r)
                container.add_data(a_key[0],a_key[1],ret)

            # j+1 is better than j
            if ret == 2:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                if rank_arr[j+1] == rank_arr[j]:
                    if j!=0:
                        if rank_arr[j-1]!=rank_arr[j]:
                            rank_arr[j+1:] = rank_arr[j+1:]+1
                    else:
                        rank_arr[j+1:] = rank_arr[j+1:]+1
                else:
                    if j!=0:
                        if rank_arr[j-1]==rank_arr[j]:
                            rank_arr[j+1:] = rank_arr[j+1:]-1

            # j+1 is as good as j
            if ret == 1:
                if rank_arr[j+1] != rank_arr[j]:
                    rank_arr[j+1] = rank_arr[j]
                    rank_arr[j+2:] = rank_arr[j+2:]-1

    #print(arr)

    arr_id = []
    for a in arr:
        arr_id.append(id_map[a])
    #print(arr_id)
    #exit(-1)
    return list(zip(arr_id,rank_arr))

# get all algorithms having a specific rank
def get_cluster(data,rank):
    cluster = []
    for r in data:
        if r[1] == rank:
            cluster.append(r[0])
    return cluster


def getF(data,T,m,r,t):
    L = []
    C = []
    for i in range(T):
        # get ranks
        ranks = bubbleSort(data,t,m,r)
        # accumulate algs with rank 1
        L = L + get_cluster(ranks,1)
    F = np.unique(L)
    for f in F:
        #compute rel score for algs in F
        C.append(float(L.count(f))/T)
    return F,C



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('-T', type=int,required=True)
    parser.add_argument('-M', type=int,required=True)
    parser.add_argument('-K', type=int, required=True)
    parser.add_argument('-t', type=float, required=True)
    parser.add_argument('-S', type=int, required=True)
    #parser.add_argument('--slrum',dest='slrum', action='store_true')
    params = parser.parse_args()
    with open(params.file) as f:
        data = json.load(f)

    S = params.S
    T = params.T
    M = params.M
    K = params.K
    t = params.t
    FxCs = {}
    for k,v in data.items():
        # if not "01" in k:
        #     continue
        data_r = reduce_sample_size(v,S)
        FxCs[k] = {}
        #FxCs[k]["F"], FxCs[k]["C"] = getF(data_r,1,S,1,0.9)
        FxCs[k]["F"], FxCs[k]["C"] = getF(data_r,T,K,M,t)
        l = list(zip(FxCs[k]["F"], FxCs[k]["C"]))
        res = sorted(l, key = operator.itemgetter(1),reverse=True)
        print(k.split("/")[-1], res)

    out_id = params.file.split("/")[-1].split(".")[0]
    out_file = "logs/scores-{}_{}_{}_{}_{}_{}.json".format(out_id,S,T,M,K,t)
    with open(out_file,"w") as f:
        json.dump(FxCs,f,cls=NpEncoder)
