import random
import shutil
import os
import sys
import json
import pprint
from linnea.examples.random_expressions import generate_equation
from linnea.derivation.graph.derivation import DerivationGraph
import linnea.examples.application as lapp
import linnea.config
import math
import application as ap
import utils

def generate_expressions():

    experiments = ARGS["experiments"]
    divide_runs = ARGS["divide_runs"]
    sub_div_list = []
    if(divide_runs > 0):
        num_divisions = math.ceil(len(experiments)/divide_runs)

        for d in range(num_divisions):
            start = d*divide_runs
            end = min((d+1)*divide_runs, len(experiments))
            sub_div_list.append(experiments[start:end])
    else:
        sub_div_list.append(experiments)


    eqns_list = []
    for i,sub_div in enumerate(sub_div_list):
        ARGS["SD"+str(i)] = sub_div
        eqns = {}
        for exp_id in sub_div:
           expr = getattr(ap, exp_id)()
           print("Expression: ", expr.eqns, "arg: ", exp_id)
           key = "sample_"+exp_id
           eqns[key] = expr.eqns
        eqns_list.append(eqns)

    return eqns_list

def generate_code(id, equations,out_dir):
    linnea.config.clear_all()
    graph = DerivationGraph(equations)
    graph.derivation(time_limit=TIME_LIMIT,
                     merging=True,
                     dead_ends=True,
                     pruning_factor=1.5)

    algorithms = graph.write_output(code=True,
                                   derivation=True,
                                   output_name=id,
                                   k_best=False,
                                   subdir_name_experiments = "experiments",
                                   experiment_code=True,
                                   algorithms_limit=100,
                                   graph=True,
                                   no_duplicates=True)

    intensity = ""
    for i,algorithm in enumerate(algorithms):
        intensity += str(i) + "\t" + str(algorithm.data) + "\t" + str(algorithm.cost) + "\t" + str(algorithm.intensity) + "\n"

    f = open(os.path.join(out_dir,id,"intensity.txt"),"w")
    f.write(intensity)
    f.close()

if __name__ == "__main__":

    try:
        with open(sys.argv[1]) as f:
            ARGS = json.load(f)
            pprint.pprint(ARGS)
    except IndexError:
        print("Error: Pass config file")
        print("Usage: python 01_generate_test_expression.py config.json")
        exit(code=-1)

    method_folder = os.path.join("logs",ARGS["expression_id"])
    if not os.path.exists(method_folder):
       os.mkdir(method_folder)

    #expression_id = ARGS["method_name"]+"_D"+str(ARGS["divide_runs"])
    expression_id = utils.get_expressions_id(ARGS)

    TIME_LIMIT = ARGS["time_limit"]
    eqns_list = generate_expressions()
    #exit(-1)

    expressions_folder = os.path.join(method_folder,expression_id)
    if os.path.exists(expressions_folder):
        print("Removing {} ... ".format(expressions_folder))
        shutil.rmtree(expressions_folder)

    os.mkdir(expressions_folder)

    with open(os.path.join(expressions_folder,"config.json"),"w") as f:
         json.dump(ARGS,f)

    for i, eqns in enumerate(eqns_list):

        out_dir = os.path.join(expressions_folder, "SD"+str(i)+"_"+expression_id,"expressions")

        for id,expr in eqns.items():
            linnea.config.set_output_code_path(out_dir)
            linnea.config.set_generate_graph(True)
            linnea.config.init()
            generate_code(id,expr,out_dir)
