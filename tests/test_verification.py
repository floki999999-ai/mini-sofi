import hashlib, json, tempfile, unittest
from pathlib import Path
import numpy as np
from mini_sofi import Network, ROOT, inspect, split_indices

class Verification(unittest.TestCase):
    def test_originals(self):
        for name,digest in json.loads((ROOT/'originals/SHA256.json').read_text()).items():
            self.assertEqual(hashlib.sha256((ROOT/'originals'/name).read_bytes()).hexdigest(),digest)

    def test_splits(self):
        a,_=inspect(ROOT/'originals/mondo.csv');parts=split_indices(a[:,2],42)
        self.assertEqual(len(np.unique(np.concatenate(parts))),len(a))
        self.assertEqual(sum(map(len,parts)),len(a))
        for x,y in zip(parts,split_indices(a[:,2],42)):
            np.testing.assert_array_equal(x,y)
            self.assertEqual(set(a[x,2]),{0.,1.})

    def test_gradients(self):
        rng=np.random.default_rng(7);x=rng.random((5,2));y=rng.integers(0,2,(5,1))
        for kind in ['baseline','communicating','isolated','monolithic']:
            m=Network(kind,42);g=m.gradients(x,y)
            for key,value in m.params.items():
                for flat in rng.choice(value.size,min(3,value.size),replace=False):
                    idx=np.unravel_index(flat,value.shape);old=value[idx];eps=1e-5
                    value[idx]=old+eps;hi=np.mean((m.forward(x)-y)**2)
                    value[idx]=old-eps;lo=np.mean((m.forward(x)-y)**2)
                    value[idx]=old
                    self.assertAlmostEqual(g[key][idx],(hi-lo)/(2*eps),places=8,msg=f'{kind}/{key}/{idx}')

    def test_checkpoint_and_reproducibility(self):
        x=np.array([[.1,.2],[.8,.9]]);y=np.array([[0.],[1.]])
        for kind in ['baseline','communicating','isolated','monolithic']:
            a,b=Network(kind,42),Network(kind,42)
            for m in [a,b]:
                for _ in range(3):
                    g=m.gradients(x,y)
                    for k in m.params:m.params[k]-=.4*g[k]
            for k in a.params:np.testing.assert_array_equal(a.params[k],b.params[k])
            with tempfile.TemporaryDirectory() as temp:
                a.save(Path(temp)/'weights.npz',epoch=3)
                restored,meta=Network.load(Path(temp)/'weights.npz')
                np.testing.assert_array_equal(a.forward(x),restored.forward(x))
                self.assertEqual(meta['epoch'],3)

    def test_isolation(self):
        x=np.array([[.2,.3],[.2,.9]])
        c,i=Network('communicating',42),Network('isolated',42)
        for key in c.params:np.testing.assert_array_equal(c.params[key],i.params[key])
        _,ct=c.forward(x,True);_,it=i.forward(x,True)
        np.testing.assert_array_equal(it[4][2][0],it[4][2][1])
        self.assertGreater(abs(float(ct[4][2][0,0]-ct[4][2][1,0])),1e-8)

if __name__=='__main__': unittest.main()
