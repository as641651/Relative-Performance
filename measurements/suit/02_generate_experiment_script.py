import pkg_resources
import os
import random
import glob
import shutil
import json
import sys
import pprint
import re
import stat
import utils

DEBUG_FLOPS = False

def generate_julia_runner(test_exp_folder,result_folder,id,sub_reps, num_threads, delta_threads):

    base_folder = os.path.join(test_exp_folder, "Julia/")
    #TEMPLATE_DICT["exp_dir"] = base_folder
    if delta_threads > 0:
        TEMPLATE_DICT["num_threads"] = random.randint(num_threads-delta_threads, num_threads)
    else:
        TEMPLATE_DICT["num_threads"] = num_threads

    operand_file = os.path.abspath(os.path.join(base_folder,"operand_generator.jl"))
    TEMPLATE_DICT["include_operand_generator"] = "include(\"{}\")".format(operand_file)

    experiment_folder = os.path.abspath(os.path.join(base_folder,"experiments"))
    algorithms = glob.glob(experiment_folder + "/*.jl")
    algorithms = [(exp.split("/")[-1].split(".")[0],exp) for exp in algorithms]
    random.shuffle(algorithms)

    include_experiment = ""
    for alg in algorithms:
        include_experiment += "include(\"{}\")\n".format(alg[1])
    TEMPLATE_DICT["include_experiments"] = include_experiment

    experiments = ""
    algorithms = algorithms*sub_reps
    for alg in algorithms:
        exp_name = alg[0]
        experiments += "\tBenchmarker.add_data(plotter, [\"{}\"; 24], Benchmarker.measure(2, {}, map(MatrixGenerator.unwrap, matrices)...) );\n".format(exp_name,exp_name)
    TEMPLATE_DICT["experiments"] = experiments

    timings_folder = os.path.abspath(os.path.join(result_folder,"timings"))
    if not os.path.exists(timings_folder):
        os.mkdir(timings_folder)
    TEMPLATE_DICT["result_file"] = os.path.join(timings_folder,"result"+str(id))

    code = RUNNER.format(**TEMPLATE_DICT)
    runner_file = os.path.join(result_folder,str(id)+"runner.jl")
    f = open(runner_file,"w")
    f.write(code)
    f.close()

    return os.path.abspath(runner_file)


def get_flops(test_cases_folder):
    test_expressions = glob.glob(test_cases_folder+"/*")
    flops = {}
    for exp in test_expressions:
        key = exp.split("/")[-1]
        flops[key] = {}
        algs = glob.glob(os.path.join(exp,"Julia/k_best/")+"/*.jl")
        #print(algs)
        for alg in algs:
            cost = 0
            with open(alg) as f:
                lines = f.readlines()
                for line in lines:
                    if "cost:" in line:
                        cost = line.split("cost:")[-1].split()[0]
                        alg_key = re.findall("\d+(?=algorithm)?$",alg.split("/")[-1].split(".")[0])[0]
                        flops[key][alg_key] = cost
                        #flops[key][alg.split("/")[-1].split(".")[0]] = cost

    return flops

def get_data(test_cases_folder):
    data = {}
    data["flops"] = {}
    data["bytes"] = {}
    data["intensity"] = {}

    test_expressions = glob.glob(test_cases_folder+"/sample*")
    for exp in test_expressions:
        key = exp.split("/")[-1]
        data["flops"][key] = {}
        data["bytes"][key] = {}
        data["intensity"][key] = {}
        with open(os.path.join(exp,"intensity.txt")) as f:
            lines = f.readlines()
            for line in lines:
                alg,bytes,cost,intensity = line.strip().split()
                data["flops"][key][alg] = cost
                data["bytes"][key][alg] = bytes
                data["intensity"][key][alg] = intensity

    return data

if __name__ == "__main__":

    try:
        with open(sys.argv[1]) as f:
            ARGV = json.load(f)
            pprint.pprint(ARGV)
    except IndexError:
        print("Error: Pass config file")
        print("Usage: python 02_generate_experiment_script.py config.json")
        exit(code=-1)


    TEMPLATE_DICT = {}
    num_threads = ARGV["num_threads"]
    delta_threads = ARGV["delta_threads"]
    SUB_REPS = 1
    NUM_REPS = ARGV["num_reps"]

    method_folder = ARGV["expression_id"]
    exp_folder = utils.get_expressions_id(ARGV)
    exp_folder_base = os.path.join("logs",method_folder,exp_folder)
    #result_folder_name = "results_"+ARGV["test_expression_id"]+"_"+str(ARGV["num_threads"])+"t_"+str(ARGV["delta_threads"])+"d_"+str(ARGV["num_reps"])+"r"
    result_folder_name = utils.get_results_id(ARGV)

    sd_expressions = glob.glob(exp_folder_base+"/SD*")
    print(sd_expressions)

    template_path = "templates/"
    RUNNER = pkg_resources.resource_string(__name__,os.path.join(template_path,"runner.jl")).decode("UTF-8")

    for sd_folder in sd_expressions:
        result_folder = os.path.join(sd_folder,result_folder_name)

        if os.path.exists(result_folder) and not DEBUG_FLOPS:
            print("Removing {} ... ".format(result_folder))
            shutil.rmtree(result_folder)


        expression_folder = os.path.join(sd_folder,"expressions")
        test_expressions = glob.glob(expression_folder+"/sample*")

        if not os.path.exists(result_folder):
            os.mkdir(result_folder)

        compute_data = get_data(expression_folder)

        with open(os.path.join(result_folder,"computeData.json"),"w") as f:
            json.dump(compute_data,f)
        if DEBUG_FLOPS:
            exit(code=-1)

        with open(os.path.join(result_folder,"config.json"),"w") as f:
            json.dump(ARGV,f)


        runners = []
        for test_exp_folder in test_expressions:
            expression = test_exp_folder.split("/")[-1]
            result_folder_exp = os.path.join(result_folder,expression)
            if not os.path.exists(result_folder_exp):
                os.mkdir(result_folder_exp)
            for i in range(NUM_REPS):
                runner_file = generate_julia_runner(test_exp_folder,result_folder_exp,i,SUB_REPS,num_threads,delta_threads)
                runners.append(runner_file)

        random.shuffle(runners)

        if ARGV["cluster"]:
            runner_prepend = "source ~/.bashrc\n"
            runner_prepend +="shopt -s expand_aliases\n"
            julia_call = "julia16 "
        else:
            runner_prepend = ""
            julia_call = "/julia/julia "

        code = runner_prepend
        for exp in runners:
           code += "echo \"Running " + exp + "\"\n"
           code += julia_call+exp + " \n"
           code += "sleep 0.2\n\n"

        overall_runner_file = os.path.join(result_folder,"runner.sh")
        f = open(overall_runner_file,"w")
        f.write(code)
        f.close()

        st = os.stat(overall_runner_file)
        os.chmod(overall_runner_file,st.st_mode | stat.S_IEXEC)
