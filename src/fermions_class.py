import numpy as np
import src.ManyBodyBasis as mbb
import src.lattice_tool as lt
from scipy.sparse import csr_matrix

def getspin(b,i):   #spin (0 or 1) at 'i' in the basis 'b'
    return (b>>i) & 1

def bitflip(b,i):  #Flips bits (1-->0, 0-->1) at loc 'i'
    return b^(1<<i)

def lcounter(b,i):  #left_counter: #1 in b left to site i
    num = b>>(i+1)
    return bin(num).count('1')

def fsign(b,imin,imax):
    n1 = lcounter(b,imin)
    n2 = lcounter(b,imax)
    sign = 0
    if getspin(b,imax)==1:
        sign = n1 + n2 - 1
    else:
        sign = n1 + n2
    return (-1)**sign

def hoping(config, imin, imax):
    occ_i = getspin(config,imin)
    occ_j = getspin(config,imax)
    newconfig = None
    if occ_i != occ_j:
        temp = config
        newconfig = bitflip(bitflip(temp,imin),imax)
    return newconfig


class Hubbard(lt.NN_bonds):
    def __init__(self,hop,Uint,nup,ndn, 
                Type,Lx,Ly,boundary_condn):

        self.t = hop
        self.U = Uint
        self.nup = nup
        self.ndn = ndn
        super().__init__(Type,Lx,Ly,boundary_condn)

    def constructMatrix(self):
        Lx = self.lx
        Ly = self.ly
        uc = len(self.unit_cell())
        L = Lx*Ly*uc
    
        nup = self.nup
        ndn = self.ndn
        t = self.t
        U = self.U

        #basis list
        configs = mbb.Basis("Hubbard",L,nup,ndn).basis()
        hilbert = len(configs)
        config_lookup = {}
        for k, config in enumerate(configs):
            config_lookup[config] = k
    
        #lattice and bonds
        lattice = self.type
        bc = self.bc
        bonds = lt.NN_bonds(lattice,Lx,Ly,bc).bonds()

        rows = []
        cols = []
        matrix = []
        
        for  m, config in enumerate(configs):
            dn, up = config

            #Diagonal matrix elements U*niup*nidn
            if U!=0:
                for k in range(L):
                    if getspin(up,k)==1==getspin(dn,k):
                        rows.append(m)
                        cols.append(m)
                        matrix.append(U)
            
            for bond in bonds:
                s1 = min(bond)-1
                s2 = max(bond)-1

                #Cup^dag Cup operation
                ket_up = hoping(up,s1,s2)

                if ket_up!=None:
                    newConfig = (dn, ket_up)
                    loc = config_lookup[newConfig]
                    sign = fsign(up,s1,s2)
                    matel = sign*(-t)

                    rows.append(m)
                    cols.append(loc)
                    matrix.append(matel)

                #Cdn^dag Cdn operation
                ket_dn = hoping(dn,s1,s2)
                if (ket_dn!=None):
                    newConfig = (ket_dn, up)
                    loc = config_lookup[newConfig]
                    sign = fsign(dn,s1,s2)
                    matel = sign*(-t)

                    rows.append(m)
                    cols.append(loc)
                    matrix.append(matel)
        
        matrix = (matrix,(rows,cols))
        matrix = csr_matrix(matrix, shape=(hilbert,hilbert))
        return matrix

def XY(config, imin, imax):
    dn, up = config
    up_imax = getspin(up,imax)
    up_imin = getspin(up,imin)
    dn_imax = getspin(dn,imax)
    dn_imin = getspin(dn,imin)
    newConfig = None
    if (up_imax==1==dn_imin):
        newUp = bitflip(bitflip(up,imin),imax)
        newDn = bitflip(bitflip(dn,imax),imin)
        newConfig = (newDn, newUp)
    if (up_imin==1==dn_imax):
        newUp = bitflip(bitflip(up,imin),imax)
        newDn = bitflip(bitflip(dn,imax),imin)
        newConfig = (newDn, newUp)

    return newConfig

def ZZ(config, i, j):
    up, dn = config
    up_i = getspin(up,i)
    up_j = getspin(up,j)
    dn_i = getspin(dn,i)
    dn_j = getspin(dn,j)
    ZiZj = 0.5*0.5*up_i*up_j
    ZiZj += -0.5*0.5*up_i*dn_j
    ZiZj += -0.5*0.5*dn_i*up_j
    ZiZj += 0.5*0.5*dn_i*dn_j

    return ZiZj


class tJ(lt.NN_bonds):
    def __init__(self,hop,Jxy,Jz,nup,ndn,
                Type,Lx,Ly,boundary_condn):

        self.t = hop
        self.Jxy = Jxy
        self.Jz = Jz
        self.nup = nup
        self.ndn = ndn
        super().__init__(Type,Lx,Ly,boundary_condn)


    def constructMatrix(self):
        Lx = self.lx
        Ly = self.ly
        uc = len(self.unit_cell())
        L = Lx*Ly*uc
    
        nup = self.nup
        ndn = self.ndn
        t = self.t
        Jxy = self.Jxy
        Jz = self.Jz

        #basis list
        configs = mbb.Basis("tJ",L,nup,ndn).basis()
        hilbert = len(configs)
        config_lookup = {}
        for k, config in enumerate(configs):
            config_lookup[config] = k
    
        #lattice and bonds
        lattice = self.type
        bc = self.bc
        bonds = lt.NN_bonds(lattice,Lx,Ly,bc).bonds()

        rows = []
        cols = []
        matrix = []
        
        for  m, config in enumerate(configs):
            dn, up = config
            #print(f'0b{up:04b}', f'0b{dn:04b}')

            for bond in bonds:
                s1 = min(bond)-1
                s2 = max(bond)-1
                
                #Heisenberg XY part
                if Jxy!=0:
                    newConfig = XY(config,s1,s2)
                    if newConfig != None:
                        loc=config_lookup[newConfig]
                        rows.append(m)
                        cols.append(loc)
                        matrix.append(0.5*Jxy)

                #Ising term
                if Jz!=0:
                    matel = Jz*ZZ(config,s1,s2)
                    rows.append(m)
                    cols.append(m)
                    matrix.append(matel)
                
                #Cup^dag Cup operation
                if t!=0:
                    ket_up = hoping(up,s1,s2)

                    if ket_up!=None:
                        if ket_up&dn==0:
                            newConfig = (dn, ket_up)
                            loc = config_lookup[newConfig]
                            sign = fsign(up,s1,s2)
                            matel = sign*(-t)

                            rows.append(m)
                            cols.append(loc)
                            matrix.append(matel)

                    #Cdn^dag Cdn operation
                    ket_dn = hoping(dn,s1,s2)

                    if ket_dn!=None:
                        if ket_dn&up==0:
                            newConfig = (ket_dn, up)
                            loc = config_lookup[newConfig]
                            sign = fsign(dn,s1,s2)
                            matel = sign*(-t)

                            rows.append(m)
                            cols.append(loc)
                            matrix.append(matel)

        matrix = (matrix,(rows,cols))
        matrix = csr_matrix(matrix, shape=(hilbert,hilbert))
        return matrix

