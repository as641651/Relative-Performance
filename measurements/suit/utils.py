import random
import math
import numpy as np


def get_expressions_id(args):
    return args["expression_id"]+"_D"+str(args["divide_runs"])

def get_results_id(args):
    return "results_"+args["arch_id"]+"_"+str(args["num_threads"])+"t_"+str(args["delta_threads"])+"d_"+str(args["num_reps"])+"r"
