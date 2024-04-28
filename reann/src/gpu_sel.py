import os

def gpu_sel(local_size):
    memory_gpu=[int(x.split()[2]) for x in open('gpu_info','r').readlines()]
    if memory_gpu:
       gpu_queue=sorted(range(len(memory_gpu)), key=lambda k: memory_gpu[k],reverse=False)
       str_queue=""
       for i in gpu_queue[:local_size]:
           str_queue+=str(i)
           str_queue+=", "
       os.environ['CUDA_VISIBLE_DEVICES']=str_queue[:-2]
       #string="export CUDA_VISIBLE_DEVICES='"+str_queue[:-2]+"'"
       #print(string)
       #os.system(string)
       print(os.environ.get('CUDA_VISIBLE_DEVICES'))
