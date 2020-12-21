# IP Project Auto Grader
This project is used -with others- to auto-grade image processing projects. It can operate on a folder that includes multiple students' submissions. For portability, it uses docker and conda environment to provide a level of flexibility. 
** Notes: docker related files shouldn't be changed and might be replaced during the final test. 
## How to use?
Your work should be inside students-work with folder structure as shown in later sections. You should change contents as following:
1. Modify main.py to run your code with the same structure and cmd arguments. 
2. Modify requirements.yml to match your requirements. 
3. Run your code through docker and check the results. Repeat this step regularly to make sure your code will work eventually.

## To export Environment: 
---

After activating the environment. Execute: 

```conda env export > requirements.yml```

Note: It might need more changes after creating it.

## Directory Structure
---
students-work/\<submitterid>/\<team-form>

** submitterid: id of sumbitter student. It is added directly by elearn submit system. For testing use any id. 

** team-form: Folder name should be SEM-## or CRD-## not Team number only.

### code-directory
---
inside the \<team-form> directoy:
There should be at least: 
1. requirements.yml: conda requirements file.
2. main.py: main python file. 
3. DockerFile: to construct docker image. Shouldn't be changed.
4. docker-compose.yml: to run docker image with parameters. Change image name only as written in the todo inside the file. 
5. All other code files. 
### Other folders
output-folder: it will conatain the output texts after executing the code. it must be outside code directory.

test-cases-folder: it should contain test cases images. it must be outside code directory.

** Don't use spaces in any directory path. 


## Grader Python:
-----------------

** All paths should be absolute in the commands and should NOT include any spaces. 

```python run-students-docker-STD.py [code-directory] [test-cases-folder] [output-folder] -v 1```
    
With down:

```python run-students-docker-STD.py [code-directory] [test-cases-folder] [output-folder] -v 1 -d 1```



## What will you deliver?

You will submit one zip file directly including one directory named [team-form] as explained above. Other directories and files shouldn't be submitted!


## Dependencies: 
* python 3.6+
* Linux
* docker-compose: [Check here](https://docs.docker.com/engine/install/ubuntu/), [here](https://docs.docker.com/engine/install/linux-postinstall/) and [here](https://docs.docker.com/compose/install/)



-- Code and test methods need more improvement and testing. Only use as a guideline. 
