"""Mini-Sofi 1.0: deterministic NumPy training and controlled communication."""
from pathlib import Path
import argparse, csv, hashlib, json, platform, sys, time, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def inspect(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] != ['x1', 'x2', 'risposta']:
        raise ValueError('Expected header x1,x2,risposta')
    if len(rows) < 11 or any(len(row) != 3 for row in rows[1:]):
        raise ValueError('Expected at least 10 data rows, each with exactly 3 columns')
    a = np.asarray(rows[1:], dtype=np.float64)
    if not np.isfinite(a).all() or not np.isin(a[:, 2], [0, 1]).all():
        raise ValueError('Non-finite values or non-binary targets')
    if len(np.unique(a[:, :2], axis=0)) != len(a):
        raise ValueError('Repeated inputs: group split required to avoid leakage')
    classes, counts = np.unique(a[:, 2], return_counts=True)
    if len(classes) != 2 or min(counts) < 7:
        raise ValueError('Both classes need at least 7 examples for stratified splitting')
    report = dict(rows=len(a), columns=3, header=rows[0], missing_values=0,
                  finite=True, min=a.min(0).tolist(), max=a.max(0).tolist(),
                  classes={str(int(k)):int(v) for k,v in zip(classes,counts)},
                  duplicate_rows=len(a)-len(np.unique(a,axis=0)),
                  sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return a, report

def split_indices(y, seed):
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for c in np.unique(y):
        ids = rng.permutation(np.flatnonzero(y == c))
        n1, n2 = int(.70*len(ids)), int(.85*len(ids))
        for dest, chunk in zip(parts, [ids[:n1], ids[n1:n2], ids[n2:]]):
            dest.extend(chunk.tolist())
    return [rng.permutation(p) for p in parts]

class Network:
    def __init__(self, kind, seed):
        self.kind, self.params = kind, {}
        rng = np.random.default_rng(seed)
        def layer(name, ni, no):
            # Original initialization, shared across communicating/isolated controls.
            self.params[name+'W'] = rng.uniform(-1, 1, (ni,no))
            self.params[name+'b'] = np.zeros((1,no))
        if kind in ('baseline','monolithic'):
            dims = [2,64,64,32,1] if kind == 'baseline' else [2,32,64,1]
            for i,(ni,no) in enumerate(zip(dims,dims[1:])): layer(str(i),ni,no)
        else:
            for name, ni, no in [('a',1,32),('b',1,32),('c',64,16),('d',64,16),('e',16,1),('f',16,1)]:
                layer(name,ni,no)

    def forward(self, x, gradients=False):
        tape=[]
        def dense(z,name):
            out=sigmoid(z @ self.params[name+'W']+self.params[name+'b'])
            tape.append((name,z,out))
            return out
        if self.kind in ('baseline','monolithic'):
            p=x
            for i in range(len(self.params)//2): p=dense(p,str(i))
        else:
            a,b=dense(x[:,:1],'a'),dense(x[:,1:],'b')
            gate=0. if self.kind=='isolated' else 1.
            c=dense(np.concatenate([a,gate*b],axis=1),'c')
            d=dense(np.concatenate([b,gate*a],axis=1),'d')
            p=(dense(c,'e')+dense(d,'f'))/2
        return (p,tape) if gradients else p

    def gradients(self,x,y):
        p,tape=self.forward(x,True)
        grad={}
        def back(item,g):
            name,z,out=item
            delta=g*out*(1-out)
            grad[name+'W']=z.T@delta
            grad[name+'b']=delta.sum(0,keepdims=True)
            return delta@self.params[name+'W'].T
        # Exact derivative of mean squared error.
        g=2*(p-y)/len(y)
        if self.kind in ('baseline','monolithic'):
            for item in reversed(tape): g=back(item,g)
        else:
            ga=back(tape[4],g/2); gb=back(tape[5],g/2)
            gc=back(tape[2],ga); gd=back(tape[3],gb)
            gate=0. if self.kind=='isolated' else 1.
            back(tape[0],gc[:,:32]+gate*gd[:,32:])
            back(tape[1],gd[:,:32]+gate*gc[:,32:])
        return grad

    def save(self,path,**metadata):
        np.savez_compressed(path,**self.params,metadata=json.dumps(dict(kind=self.kind,**metadata)))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']))
            model=cls(meta['kind'],0)
            model.params={k:z[k].copy() for k in model.params}
        return model,meta

def metrics(y,p):
    truth=y.ravel().astype(int); pred=(p.ravel()>=.5).astype(int)
    tp=int(((truth==1)&(pred==1)).sum()); tn=int(((truth==0)&(pred==0)).sum())
    fp=int(((truth==0)&(pred==1)).sum()); fn=int(((truth==1)&(pred==0)).sum())
    precision=tp/(tp+fp) if tp+fp else 0.
    recall=tp/(tp+fn) if tp+fn else 0.
    return dict(mse=float(np.mean((y-p)**2)),accuracy=float(np.mean(truth==pred)),
                precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
                balanced_accuracy=(recall+tn/(tn+fp))/2,confusion_matrix=[[tn,fp],[fn,tp]])

def run(args):
    if args.epochs < 1 or (args.epochs > 500 and not args.allow_long_run):
        raise ValueError('Runs above 500 epochs require --allow-long-run after user confirmation.')
    if args.eval_every < 1:
        raise ValueError('eval-every must be positive')
    if args.lr<=0 or not np.isfinite(args.lr): raise ValueError('lr must be finite and positive')
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    a,report=inspect(args.data)
    train,val,test=split_indices(a[:,2],args.seed)
    np.savez_compressed(out/'splits.npz',train=train,validation=val,test=test)
    (out/'dataset_report.json').write_text(json.dumps(report,indent=2))
    config={**vars(args),'python':sys.version,'numpy':np.__version__,'matplotlib':matplotlib.__version__,'platform':platform.platform(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'thread_environment':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
            'split_counts':dict(train=len(train),validation=len(val),test=len(test)),
            'selection':'minimum validation MSE; test evaluated once after selection',
            'loss':'MSE','threshold':.5,'normalization':'none; same raw inputs as original'}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    models=['baseline'] if args.mode=='baseline' else ['communicating','isolated','monolithic']
    summary={}
    for kind in models:
        model=Network(kind,args.seed)
        x,y=a[:,:2],a[:,2:]
        history=[]; best=float('inf'); best_epoch=0
        started=time.monotonic()
        # Include initialization; no test access during fitting or selection.
        for epoch in range(args.epochs+1):
            if epoch:
                grads=model.gradients(x[train],y[train])
                for k in model.params: model.params[k]-=args.lr*grads[k]
            if epoch % args.eval_every and epoch != args.epochs:
                continue
            tm,vm=metrics(y[train],model.forward(x[train])),metrics(y[val],model.forward(x[val]))
            if not np.isfinite(tm['mse']): raise FloatingPointError('Non-finite loss')
            history.append(dict(epoch=epoch,train_loss=tm['mse'],validation_loss=vm['mse'],train_accuracy=tm['accuracy'],validation_accuracy=vm['accuracy']))
            if vm['mse']<best:
                best,best_epoch=vm['mse'],epoch
                model.save(out/f'{kind}_best.npz',epoch=epoch,seed=args.seed,data_sha256=report['sha256'])
            if epoch % 1000 == 0 or epoch == args.epochs:
                model.save(out/f'{kind}_latest.npz',epoch=epoch,seed=args.seed,data_sha256=report['sha256'])
            progress=dict(model=kind,epoch=epoch,total_epochs=args.epochs,train_mse=tm['mse'],validation_mse=vm['mse'],best_epoch=best_epoch,elapsed_seconds=time.monotonic()-started,status='training')
            temp=out/'progress.tmp'
            temp.write_text(json.dumps(progress,indent=2))
            temp.replace(out/'progress.json')
            with (out/f'{kind}_history.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=history[0]);writer.writeheader();writer.writerows(history)
            print(json.dumps(progress),flush=True)
        model.save(out/f'{kind}_last.npz',epoch=args.epochs,seed=args.seed,data_sha256=report['sha256'])
        selected,meta=Network.load(out/f'{kind}_best.npz')
        result=dict(parameters=sum(v.size for v in model.params.values()),best_epoch=best_epoch,
                    train=metrics(y[train],selected.forward(x[train])),validation=metrics(y[val],selected.forward(x[val])),
                    test=metrics(y[test],selected.forward(x[test])))
        summary[kind]=result
        with (out/f'{kind}_history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        for split in ['train','validation']:
            axs[0].plot([h['epoch'] for h in history],[h[split+'_loss'] for h in history],label=split)
            axs[1].plot([h['epoch'] for h in history],[h[split+'_accuracy'] for h in history],label=split)
        for ax,title in zip(axs,['MSE','Accuracy']): ax.set(xlabel='Epoch',ylabel=title);ax.legend();ax.grid(alpha=.2)
        fig.suptitle(kind);fig.tight_layout();fig.savefig(out/f'{kind}_curves.png',dpi=140);plt.close(fig)
    (out/'metrics.json').write_text(json.dumps(summary,indent=2))
    (out/'progress.json').write_text(json.dumps(dict(status='completed',epochs=args.epochs,models=models),indent=2))
    print(json.dumps({'dataset':report,'splits':config['split_counts'],'results':summary},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default=str(ROOT/'originals/mondo.csv'))
    p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=['baseline','experiment'],default='baseline')
    p.add_argument('--allow-long-run',action='store_true',help='Explicit opt-in after user confirmation')
    p.add_argument('--eval-every',type=int,default=1,help='Evaluate validation and select checkpoint every N epochs, plus initial/final')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--lr',type=float,default=.4,help='MSE exact derivative; .4 matches original .8 update factor')
    run(p.parse_args())
