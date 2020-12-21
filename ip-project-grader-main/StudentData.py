import os

class StudentData:
    def __init__(self, line,team,submitter_directory_full_path,student_directory, args):
        
        self.line = line
        self.team = team
        self.env_name = f'{line}{team}'
        
        # self.req_path = req_path
        self.submitter_directory_full_path = submitter_directory_full_path
        self.student_directory = student_directory
        self.args = args
    def getCodePath(self):
        path = f'{self.submitter_directory_full_path}/{self.student_directory}'
        return path
    def getCreateEnvCMD(self):
        req_path = f'{self.submitter_directory_full_path}/{self.student_directory}/requirements.yml'
        return f'conda env create --name {self.env_name} --file={req_path}'
    def getRemoveEnvCMD(self):
        return f'conda remove --name {self.env_name} --all -y'
    def getActivateEnv(self):
        return f'conda activate {self.env_name}'

    def getTestCMD(self):
        python_file = f'{self.submitter_directory_full_path}/{self.student_directory}/main.py'
        return f'python {python_file} {self.args.testcases} {self.args.outputfolder}'
    def createGetDir(self):
        if not os.path.exists(self.args.outputfolder):
            os.makedirs(self.args.outputfolder)
        complete_path = f'{self.args.outputfolder}/{self.env_name}'
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)            
        return complete_path
    def getUserContents(self):
        user_arr = {}
        user_out_dir = self.createGetDir()
        # print(f'user_out_dir: {user_out_dir}')
        for user_file in os.listdir(user_out_dir):
            # print(f'{user_file}')
            with open(f'{user_out_dir}/{user_file}') as f:
                read_data = f.read().replace(" ", "").replace("\n", "")
                # print(f'read_data: {read_data}')
                # break
                user_arr[f'{user_file}'] = read_data
        return user_arr

    def getDockerComposeDownCMD(self):
        docker_down_cmd = f'INPUT_TEST={self.args.testcases} OUTPUT_TEST={self.args.outputfolder}/{self.env_name} docker-compose down --rmi all -v --remove-orphans'
        return docker_down_cmd
    def getDockerComposeCMD(self):
        docker_build_cmd = f'INPUT_TEST={self.args.testcases} OUTPUT_TEST={self.args.outputfolder}/{self.env_name} docker-compose build '
        docker_up_cmd = f'INPUT_TEST={self.args.testcases} OUTPUT_TEST={self.args.outputfolder}/{self.env_name} docker-compose up'


        # --exit-code-from ipproject


        docker_build_up_cmd = f'INPUT_TEST={self.args.testcases} OUTPUT_TEST={self.args.outputfolder}/{self.env_name} docker-compose up --build --force-recreate --abort-on-container-exit --exit-code-from ipproject'




        # return [docker_build_cmd,docker_up_cmd]
        return docker_build_up_cmd


 
