import json
import pprint
import os
import sys
import stat
import pathlib
import suit.utils as utils
import math
import shutil
import pkg_resources
import argparse

def aices2_anynode():
    code = "#SBATCH -A aices2\n"
    code += "#SBATCH --cores-per-socket={cores_per_socket}\n"
    code += "#SBATCH --exclusive\n"
    return code

def linuxihdc072():
    code = "#SBATCH -A aices2\n"
    code += "#SBATCH --nodelist=linuxihdc072\n"
    code += "#SBATCH --exclusive\n"
    return code


def slrum_script_code(node_list,job_name,num_hours,mem,cores,commands):
    template_path = "suit/templates/"
    script = pkg_resources.resource_string(__name__,os.path.join(template_path,"slrum_script.sh")).decode("UTF-8")

    template = {}
    template["node"] = globals()[node_list]()
    template["job_name"] = job_name
    template["memory"] = mem
    template["num_hours"] = num_hours
    template["cores"] = cores
    template["commands"] = commands

    code = script.format(**template)
    return code



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('--slrum',dest='slrum', action='store_true')
    params = parser.parse_args()

    isSlrum = params.slrum
    config_file = os.path.abspath(params.file)
    if not os.path.exists(config_file):
      print("config file does not exists")
      exit(code=-1)

    with open(config_file) as f:
      ARGS = json.load(f)
      pprint.pprint(ARGS)

    script_file_dir = os.path.join(os.path.dirname(config_file),"scripts")
    if not os.path.exists(script_file_dir):
    #    shutil.rmtree(script_file_dir)
        os.mkdir(script_file_dir)

    this_script_directory = pathlib.Path(__file__).parent.absolute()
    suit_directory = os.path.join(this_script_directory,"suit")


    seed = ARGS["seed"]
    divide_runs = ARGS["divide_runs"]
    experiments_list = ARGS["experiments"]
    num_divisions = 1
    if(divide_runs > 0):
        num_divisions = math.ceil(len(experiments_list)/divide_runs)

    method_folder = ARGS["expression_id"]
    exp_folder = utils.get_expressions_id(ARGS)
    exp_folder_base = os.path.join("logs",method_folder,exp_folder)
    result_folder_name = utils.get_results_id(ARGS)

    gen_code = ""
    if not ARGS["reuse_expressions"]:
        gen_code = "python3 01_generate_test_expressions.py {}\n".format(config_file)

    for div in range(num_divisions):
        code = "cd {}\n".format(suit_directory)
        if div==0:
            code += gen_code
            code += "python3 02_generate_experiment_script.py {}\n".format(config_file)

        sd_folder = os.path.join(exp_folder_base,"SD"+str(div)+"_"+exp_folder)
        results_folder = os.path.join(sd_folder,result_folder_name)
        #print("Results folder : ", results_folder)

        code += "cd {}\n".format(results_folder)
        code += "./runner.sh\n"

        code += "cd {}/\n".format(suit_directory)
        code += "python3 03_gather_results.py {}\n".format(results_folder)
        #code += "python3 04_analyse.py {}\n".format(results_folder)

        script_file = "SD"+str(div)+"_measure_"+config_file.split("/")[-1].split(".json")[0]+".sh"
        if isSlrum:
            script_file = "Sl_"+script_file
            memory = ARGS["memory"]
            num_hours = ARGS["num_hours"]
            job_name = script_file
            cores = ARGS["num_threads"]
            node = ARGS["node"]
            code = slrum_script_code(node,job_name, num_hours, memory, cores, code)

        script_file = os.path.join(script_file_dir,script_file)
        #print(script_file)
        #exit(-1)
        f = open(script_file,"w")
        f.write(code)
        f.close()

        st = os.stat(script_file)
        os.chmod(script_file,st.st_mode | stat.S_IEXEC)
