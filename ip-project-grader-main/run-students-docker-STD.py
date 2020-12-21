# Python program to demonstrate
# command line arguments
 
 
import argparse
import os
from StudentData import StudentData
from cd import cd
import subprocess
import csv
from constants import *
import pickle
# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("std", help = "Students Folder")
parser.add_argument("testcases", help = "Test Cases Folder")
parser.add_argument("outputfolder", help = "Output Folder")
parser.add_argument("-v", "--verbose", help = "Verbose")
parser.add_argument("-d", "--down", help = "Down Command before creating new docker")

# Read arguments from command line
args = parser.parse_args()
verbose = False
if args.verbose:
    verbose = True
newenv = False
# if args.newenv:
#     newenv = True



stds =[]
failed_folders = []
## Install environments for all 
stds_directory = args.std
testcases_directory = args.testcases
if verbose:
    print("Student Directory: % s" % args.std)
    print("Test Cases Directory: % s" % args.testcases)
for submitter_id in os.listdir(stds_directory):
    submitter_directory_full_path = os.path.join(stds_directory, submitter_id)
    if verbose:
        print("Submitter Directory: % s" % submitter_id)
    if os.path.isdir(submitter_directory_full_path):
        
        for scount, student_directory in enumerate(os.listdir(submitter_directory_full_path)):
            if verbose:
                print("Student Directory Name: % s" % student_directory)            
            try:
                [line, team] = student_directory.split('-')
                # print('line:', line)
                # print([line, team])
                if (line not in ['SEM','CRD']):
                    raise Exception("Line should  be SEM or CRD")
                std = StudentData(line,team,submitter_directory_full_path,student_directory,args)
                stds.append(std)
                break
            except Exception as e:
                failed_folders.append(submitter_directory_full_path)
                print('Couldn\'t parse team directory:', e)
if verbose:
    print('The following folders failed:',failed_folders)

for s in stds:
    if newenv:
        cmd = std.getCreateEnvCMD()
        remove_cmd = std.getRemoveEnvCMD()
        test_cmd = std.getTestCMD()
        if verbose:
            print('cmd: ',cmd)
            print('remove_cmd: ',remove_cmd)
            print('test_cmd: ',test_cmd)





# Status 0 means no error. 
stds_docker_status = {}
stds_docker_status_headers = ['Line', 'T#', 'Environment Name', 'Status']
for s in stds:
    
        
    docker_build_up_cmd = s.getDockerComposeCMD()
    docker_down_cmd = s.getDockerComposeDownCMD()
    print('docker_build_cmd: ', docker_build_up_cmd)
    with cd(s.getCodePath()):
        if args.down:
            print(f'Down Docker compose with all resources of {s.student_directory}')
            os.system(docker_down_cmd)
            
        print(f'Executing Docker compose of {s.student_directory}')
        returned_value = os.system(docker_build_up_cmd)
        print('RETURNED STATUS = ', returned_value)
        stds_docker_status[s.env_name] = [s.line, s.team, s.env_name, returned_value]
print(f'stds_docker_status: {stds_docker_status}')

status_file_path = STATUS_FILE_PATH

## Write csv
with open(status_file_path, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(stds_docker_status_headers)
    for std_env in stds_docker_status:
        spamwriter.writerow(stds_docker_status[std_env])
    


        

        

        



