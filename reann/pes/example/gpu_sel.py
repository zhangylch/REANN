import os
import sys
def gpu_sel():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >gpu_info')
    memory_gpu=[int(x.split()[2]) for x in open('gpu_info','r').readlines()]
    if memory_gpu:
        gpu_queue=sorted(range(len(memory_gpu)), key=lambda k: memory_gpu[k],reverse=True)
        str_queue=""
        for i in gpu_queue:
            str_queue+=str(i)
            str_queue+=", "
        os.environ['CUDA_VISIBLE_DEVICES']=str_queue
        os.system('rm gpu_info')
    
