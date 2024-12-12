import time
import threading
from datetime import timedelta
count=0


factor=10000
def time_stuff():
    count=0
    now=time.time()
    start= int(now*factor)
    
    while start %10!=0:
        count+=1
        now=time.time()
        start= int(now*factor)
    print(f"count {count} now {now}")

def patient_print(start_event,step,n=5,name="thread"):
    start_event.wait()
    start=time.time()
    while n>0:
        now=time.time()
        if now>=start:
            start+=step
            print(f"{name} {n} {now}")
            n-=1

    
start_event = threading.Event()
step=0.1
for k in range(4):
    name=f"thread_{k}"
    
    threading.Thread(target=patient_print,args=(start_event,step,5,name)).start()
    #threading.Thread(target=time_stuff,).start()

time.sleep(2)
start_event.set()