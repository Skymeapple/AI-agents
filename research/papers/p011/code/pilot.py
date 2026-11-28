import sys,time,json,cotlab as C
reg,eta,de,task,L,steps=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]),sys.argv[4],int(sys.argv[5]),int(sys.argv[6])
t=time.time(); m,Dtr,Dte,h=C.train(task=task,regime=reg,eta=eta,delta=de,L=L,d=64,bs=256,steps=steps,log_every=steps//4,eval_fn=C.evaluate)
r=C.evaluate(m,Dte,23,task); print(reg,eta,de,task,L,round(time.time()-t,1),{k:round(v,3) for k,v in r.items()}, [ (x['step'],round(x['acc_clean'],2),round(x['acc_trunc'],2),round(x['follow'],2)) for x in h],flush=True)
