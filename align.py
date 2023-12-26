import numpy as np



def simComplex(a,b):
    D=a.shape[0]
    return np.sum(np.conj(a)*b)/D



def genRandComplexVec(D,seed=None):
    if seed is not None:
        np.random.seed()
    
    vec=np.random.uniform(-np.pi,np.pi,size=D)
    expvec=np.exp(1j*vec)
    return expvec,vec


def get_bases(D=4000):
    bases=np.zeros((4,D))
    for i in range(4):
        _,bases[i]=genRandComplexVec(D)

    _,permvec=genRandComplexVec(D)

    return bases,permvec


def quantizecomplx(z,N):
    
    z=np.round(z,8)

    if z==0:
        return z
    
    z=z/np.abs(z)
    angle=np.angle(z)
    
    if angle<0:
        angle=2*np.pi+angle
        
    if angle>=2*np.pi:
        angle=angle-2*np.pi
        

    n=int(angle/(2*np.pi/N))
    
    if angle-2*np.pi*n/N<2*np.pi*(n+1)/N-angle:
        return np.exp(1j*2*np.pi*n/N)
    else:
        return np.exp(1j*2*np.pi*(n+1)/N)
    
    
    

def quantizeexpvec(v,N):
    D=v.shape[0]
    for d in range(D):
        v[d]=quantizecomplx(v[d],N)
    return v



def encodeQuantizeComplex(seq,basis,pos_set,scale,N):
    
    D=pos_set.shape[0]
    L=len(seq)

    basis_set=basis[seq,:]
    
    loc_bind_basis=pos_set+basis_set
    expvec=np.sum(np.exp(1j*loc_bind_basis),axis=0)

    quantexpvec=quantizeexpvec(expvec,N)

    return quantexpvec,expvec
    

def encodeComplex(seq,basis,pos_set,scale):
    D=pos_set.shape[0]
    L=len(seq)
    
    expvec=np.zeros(D,dtype=complex)
    basis_set=basis[seq]
    
    loc_bind_basis=pos_set+basis_set
    expvec=np.sum(np.exp(1j*loc_bind_basis),axis=0)

    
    #posvec=np.zeros(D)
    #for l in range(L):
    #    expvec+=np.exp(1j*(posvec+basis[int(seq[l])]))
    #    posvec+=(permvec/scale)
        
    return expvec

def quantizeComplex(a):
    D=a.shape[0]
    ret=np.zeros(D,dtype=int)
    
    for d in range(D):
        if np.real(a[d])<0:
            ret[d]=-1
        else:
            ret[d]=1
    #a=a.astype(int)
    return ret

def genEncoding(seq,bases,pos_set,scale,N=None):

    
    D=pos_set.shape[0]
    L=len(seq)
    
    if N is None:
        a=encodeComplex(seq,bases,pos_set,scale)
        return a/simComplex(a,a)**0.5
    elif N=="phase":
        a=encodeComplex(seq,bases,pos_set,scale)
        for d in range(D):
            a[d]=a[d]/np.abs(a[d])
        return a
    
    elif N==2:
        return quantizeComplex(encodeComplex(seq,bases,pos_set,scale))
    
    else:
        a,_=encodeQuantizeComplex(seq,bases,pos_set,scale,N)
        return a
    

def get_align_sim(seq1,seq2,bases,permvec,scale,split_len=10,N=None):
    dna_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq1 = np.array([dna_to_int[x] for x in seq1[0]])
    seq2 = np.array([dna_to_int[x] for x in seq2[0]])

    L=len(seq1)
    D=permvec.shape[0]
    pos_set=np.zeros((L,D))
    for l in range(L):
        pos_set[l,:]=permvec/scale*l

    sim_scores=[]

    for l in range(L//split_len):

        start=l*split_len
        end=(l+1)*split_len
        if end>L:
            end=L

        a1=genEncoding(seq1[start:end],bases,pos_set[start:end],scale,N)
        a2=genEncoding(seq2[start:end],bases,pos_set[start:end],scale,N)

        sim_scores.append(simComplex(a1,a2))

    return np.mean(sim_scores)

    
