import subprocess 
import os
import shutil
import tempfile
import time
from multiprocessing import Pool
import os
import subprocess
import numpy as np
import tempfile
import shutil

class MyModel2:
    def __init__(self, noms_params, Re, rho, source_dir=None):
        self.noms_params = noms_params
        self.Re = Re
        self.rho = rho
        self.source_dir = source_dir

    def setup_temp_dir(self):
        # Create a temporary directory and copy the content of source_dir into it
        self.temp_dir = tempfile.mkdtemp()
        if self.source_dir:
            self.copy_source_to_temp()
        print(self.temp_dir)

    def copy_source_to_temp(self):
        if self.source_dir and os.path.isdir(self.source_dir):
            for item in os.listdir(self.source_dir):
                s = os.path.join(self.source_dir, item)
                d = os.path.join(self.temp_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)

    def write_params(self, params):
        list_params = [(self.noms_params[i], params[i]) for i in range(len(params))]
        filename = os.path.join(self.temp_dir, 'Param_test.txt')
        with open(filename, 'w') as file:
            for param, value in list_params:
                file.write(f"{param} {value}\n")
        file_path = os.path.join(self.temp_dir, 'Param_evol.txt')
        with open(file_path, 'a') as file:
            file.write('-------\n')
            for param, value in list_params:
                file.write(f"{param} {value}\n")

    def read_output(self):
        file_path = os.path.join(self.temp_dir, 'res_const.txt')
        vm_values = []
        start_reading = False
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if "VMIS" in line:
                    start_reading = True
                if start_reading:
                    if '=====>' in line:
                        break
                    try:
                        vm = line.split("|")
                        if len(vm) <= 5:
                            break
                        else:
                            vm = vm[4]
                            vm = float(vm)
                            vm_values.append(vm)
                    except ValueError:
                        continue
        return vm_values

    def launch_calcul(self):
        file_path = str(self.temp_dir)
        old_sequence_str = '\\'
        new_sequence_str = '/'
        file_path = file_path.replace(old_sequence_str, new_sequence_str)
        old_sequence_str = 'C:/'
        new_sequence_str = '/mnt/c/'
        file_path = file_path.replace(old_sequence_str, new_sequence_str)
        cmd = f'wsl -d smeca /opt/sm/bin/as_run {file_path + "/export"}'
        l1 = subprocess.check_output(cmd, shell=True)
        return l1

    def launch_mesh(self):
        file_path = str(self.temp_dir)
        old_sequence_str = '\\'
        new_sequence_str = '/'
        file_path1 = file_path.replace(old_sequence_str, new_sequence_str)
        file_path2 = file_path1 + '/scirpt1.py'
        old_sequence_str = 'C:/'
        new_sequence_str = '/mnt/c/'
        file_path = file_path1.replace(old_sequence_str, new_sequence_str)
        with open(file_path2, 'r') as file:
            lines = file.readlines()
        # Modify some lines in the script for meshing
        for i in range(len(lines)):
            if '/mnt/c/Users/TOUGERON/Documents/PRO/SMECA/Modele_def/Table2_Files/RunCase_2/Result-Stage_1/_ExportedFromSalomeObject_0_1_3_3.med' in lines[i]:
                lines[i] = lines[i].replace('/mnt/c/Users/TOUGERON/Documents/PRO/SMECA/Modele_def/Table2_Files/RunCase_2/Result-Stage_1/_ExportedFromSalomeObject_0_1_3_3.med', file_path + '/_ExportedFromSalomeObject_0_1_3_3.med')
            if '/mnt/c/Users/TOUGERON/Documents/PRO/SMECA/ModeleTable_OCTAVE/Param_test.txt' in lines[i]:
                lines[i] = lines[i].replace('/mnt/c/Users/TOUGERON/Documents/PRO/SMECA/ModeleTable_OCTAVE/Param_test.txt', file_path + '/Param_test.txt')
        with open(file_path2, 'w') as file:
            file.writelines(lines)
        print(self.temp_dir)
        # Running smeca
        cmd = f'wsl -d smeca /opt/sm/bin/salome_meca python {file_path + '/scirpt1.py'}'
        process = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT)
        # Sleeping to let the process terminate
        time.sleep(25)
        # Closing all instances of smeca
        cmd_down = 'wsl --shutdown'
        subprocess.run(cmd_down, shell=True, check=True)
        return process.returncode

    def read_flamb(self):
        file_path = os.path.join(self.temp_dir, 'resu_flamb')
        with open(file_path, 'r') as file:
            lines = file.readlines()
        char_crit_values = []
        for line in lines:
            line = line.strip()
            if "=====>" in line:
                break
            if line and not any(c.isalpha() for c in line.split()[1:]):
                try:
                    char_crit = line.split()[1].replace('D', 'E')
                    char_crit = float(char_crit)
                    char_crit_values.append(char_crit)
                except (IndexError, ValueError):
                    continue
        char_crit_array = np.array(char_crit_values)
        return char_crit_array

    def read_mass(self):
        file_path = os.path.join(self.temp_dir, 'masse.txt')
        with open(file_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            if line.startswith("Table"):
                columns = line.split()
                try:
                    mass = float(columns[2])
                    #file_path = 'C:/Users/TOUGERON/Documents/PRO/SMECA/Modele_def/EVOL/OBJ/obj_val.txt'
                    #with open(file_path, 'a') as file:
                    #    file.write('-------\n')
                    #    file.write(f"{mass}\n")
                    return mass
                except (IndexError, ValueError) as e:
                    raise ValueError(f"Erreur lors de l'extraction de la masse : {e}")
        raise ValueError("La ligne contenant la masse n'a pas été trouvée.")

    def perform_calculations(self, params):
        self.setup_temp_dir()
        self.write_params(params)
        self.launch_mesh()
        self.launch_calcul()
        self.results_vm = self.read_output()
        self.results_flamb = self.read_flamb()
        self.mass = self.read_mass()
        #self.cleanup()

    def objective(self, params):
        self.perform_calculations(params)
        return self.mass

    def constr_flamb(self, params):
        self.perform_calculations(params)
        coeff = np.abs(self.results_flamb)
        return min(coeff) - 1

    def constr_vm(self, params):
        self.perform_calculations(params)
        coeff = np.abs(self.results_vm)
        return -max(coeff) + self.Re

    def cleanup(self):
        shutil.rmtree(self.temp_dir)


import otwrapy
import openturns as ot
import numpy as np

#if __name__ == '__main__':
source_directory = 'C:/Users/TOUGERON/Documents/PRO/SMECA/Modele_def/Table2_Files/RunCase_3/Result-Stage_1'
noms_params = ['L1', 'L2', 'H', 'P']
Re = 236
rho = 7800

model = MyModel2(noms_params, Re, rho, source_dir=source_directory)
def objective_function(params):
    return [model.objective(params)]

params = np.array([[2000, 500, 300, 50],[1200,200,600,70],[1600,200,600,70]]) # testing with 3 different core
number_of_calculs = 3
from multiprocessing import Pool
time0 = time.time()
if __name__ == '__main__':
    with Pool(number_of_calculs) as p:
        print(p.map(objective_function, params))
    print(time.time()-time0)
