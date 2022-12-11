from scipy.sparse.linalg import eigsh
import src.fermions_class as Fermi

#model and parameters
model = "tJ"
t = 1.0         # Hoping
Jxy = 1.0       #XY interaction
Jz = 1.0        # Ising interaction

#Lattice and system size 
LatType = "triangle"
uc = 1          #sites in unit cell
Lx = 3          #along x
Ly = 3          #along y
L = Lx*Ly*uc    #total sites
bc = 'periodic' #boundary condition

#Number of particles
Nup = L-2
Ndn = 1

H = Fermi.tJ(t,Jxy,Jz,Nup,Ndn,LatType,Lx,Ly,bc)
H = H.constructMatrix() 
#computing ground state
eigs,vecs = eigsh(H,k=1,which='SA',sigma=None)
#k: number of eigen values and eigen vectors desired
#SA: Smallest algebraic eigen value 
#sigma: Find eigenvalues near sigma using shift-invert mode

#import numpy
#eigs, vecs = numpy.linalg.eigh(H.toarray())
#Numpy does not take sparse matrix firm. Convert to array first.
print(eigs)
#print(vecs)
