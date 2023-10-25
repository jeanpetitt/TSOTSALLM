# Run a Job on PlaFRIM documentation

<!-- use SSH key and connecte -->

### Login on PlaFRIM through ssh key

open the terminal and run this command

```
ssh your_username
```


<!-- ### Request for A100 GPU

to run your project using a A100 GPU. 
launch this command:
```
    salloc -C 'sirocco&A100'
```

or 
```
    salloc -C 'sirocco22'
```

or you can using srun without salloc

```
    srun --exclusive -C sirocco --pty bash -i
```
```
    module load compiler/cuda
```

```
    nvidia-smi
``` -->

### Upgrade python version
by default the python version install on your computer is 2.x. to upgrade this version you can use guix to install software package like python.

```
guix install python python-numpy python-scipy
```

verify python version installed
```
python3 --version
```
Viewing installed software
```
guix package --list-installed
```

### Clone your project on github
```
git clone https://github.com/jiofidelus/TSOTSALLM 
```
navigate inside of the project

```
cd TSOTSALLM
```

### create virtual environment 

```
pip install virtualenv
```

```
python3 -m venv env
```

### Activate virtual environment
```
source env/bin/activate
```

### Install requirements.txt
```
pip install -r toy_submission/llama_recipes/fast_api_requirements.tx
```

### Job in Script file

create our train_job.sh file
```
touch train_job.sh
```

paste these instructions inside
```
#!/bin/bash
#SBATCH --time=01:00:00  # execution time of the job
#SBATCH --job-name=Tsotsallm  # name of the Job
#SBATCH --output=tsotsallm_output.txt  # output file
#SBATCH --partition=sirocco  # Node "sirocco"
#SBATCH --gres=gpu:a100:1  # using A100 GPU

# start training
python toy_submission/llama_recipes/train.py --model-name NousResearch/Llama-2-7b-hf --hf_rep yvelos/Tsotsallm --fine-tuned-model-name Tsotsallm  --epochs 1 --bf16 --split train
```
### Submit the Job
 
```
sbatch train_job.sh
```

### monitoring Job
after the job has submited you can control the state of the Job. by using thes command
  * List job
```
squeue
```

  * print detail of specific job
  
```
scontrol show job job_id
```