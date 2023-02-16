import numpy as np
f2=open('train/configuration','w')
f3=open('test/configuration','w')
with open('configuration','r') as f1:
    while True:
        string=f1.readline()
        if not string: break
        tmp=np.random.uniform()
        if tmp>0.9: 
           f3.write(string)
           while True:
               string=f1.readline()
               if "abprop" in string:
                  f3.write(string)
                  break
               f3.write(string)
        else:
           f2.write(string)
           while True:
               string=f1.readline()
               if "abprop" in string:
                   f2.write(string)
                   break
               f2.write(string)
f2.close()
f3.close()            
