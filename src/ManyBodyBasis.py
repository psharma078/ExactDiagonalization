import numpy as np
import copy

def convertTOdecimal(test_list):
    #res = 0
    #for ele in test_list:
    #    res = (res << 1) | ele
    res = int("".join(str(x) for x in test_list), 2)
    return res

def basisPerm(N,k,ud): #N=sites, k=number of '1' if ud= 1
                       #k=number of '0' if ud= 0
   basis = []
   state = 0
   if ud == 1:
       state = np.ones(N,dtype=int)
   else:
       state = np.zeros(N,dtype=int)

   if k==0:
       temp = np.zeros(N,dtype=int)
       basis.append(convertTOdecimal(temp))

   if k==N:
       temp = np.ones(N,dtype=int)
       basis.append(convertTOdecimal(temp))

   if k==N-1:
       for i in range(N):
           temp = copy.deepcopy(state)
           if ud == 1: temp[i] = 0
           else: temp[i] = 1
           #print(temp, convertTOdecimal(temp))
           basis.append(convertTOdecimal(temp))
   if k==N-2:
       for i in range(N):
           temp1 = copy.deepcopy(state)
           if ud == 1: temp1[i] = 0
           else: temp1[i] = 1

           for j in range(i,N):
               if j!=i:
                   temp2 = copy.deepcopy(temp1)
                   if ud==1: temp2[j] = 0
                   else: temp2[j] = 1
                   basis.append(convertTOdecimal(temp2))

   if k==N-3:
       for i in range(N):
           temp1 = copy.deepcopy(state)
           if ud == 1: temp1[i] = 0
           else: temp1[i] = 1

           for j in range(i,N):
               temp2 = copy.deepcopy(temp1)
               if j!=i:
                   if ud==1: temp2[j] = 0
                   else: temp2[j] = 1

                   for l in range(j,N):
                       if l!=j:
                           temp3 = copy.deepcopy(temp2)
                           if ud==1: temp3[l] = 0
                           else: temp3[l] = 1
                           basis.append(convertTOdecimal(temp3))
   return basis


def getspin(b,i):       #spin (0 or 1) at site 'i' in the basis 'b'
    return (b>>i) & 1

#This Basis class is contructed for nup>N-4 and ndn<4
class Basis:
    def __init__(self,model:str,Nsites:int,Nup,Ndn):
        self.N = Nsites
        self.nup = Nup
        self.ndn = Ndn
        self.model = model

    def basis(self): #This works for Nup>N-4 and Ndn<4 
        Ns = self.N
        configUp = []
        if (self.nup==0 or self.nup > Ns-4):
            configUp = basisPerm(Ns,self.nup,1)
        if 0<self.nup < 4:
            configUp = basisPerm(Ns,Ns-self.nup,0)

        configDn = []
        if (self.ndn>0 and self.ndn<4): 
            configDn = basisPerm(Ns,Ns-self.ndn,0)
        else: configDn = basisPerm(Ns,self.ndn,1)

        configs = []

        if self.model == 'Hubbard':#no constraints
            for up in configUp:
                for dn in configDn:
                    config = (dn,up)
                    configs.append(config)

        if self.model == 'tJ': #nPr/(ndn!+hole!) where r = ndn+hole
            for up in configUp:
                for dn in configDn:
                    #print(f'0b{up:09b}', f'0b{dn:09b}', up&dn)
                    if up&dn == 0:
                        configs.append((dn,up))
                    
        return configs


if __name__=='__main__':
   
    N = 9
    ba = Basis("tJ",N,0,3)
    c = ba.basis()
    for cc in c:
        dn, up = cc
        print(f'0b{dn:09b}', f'0b{up:09b}')
    print(len(c))

