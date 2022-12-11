import numpy as np

class lattice:
    def __init__(self, Type:str, Lx:int, Ly:int):
        assert Lx > 0, f"length {Lx} along a1 can't be negative!"
        assert Ly > 0, f"length {Ly} along a2 can't be negative!"
        self.type = Type
        self.lx = Lx
        self.ly = Ly

    def translation_vecs(self):
        if self.type == "honeycomb":
            a1 = np.array([3.0/2.0,np.sqrt(3)/2.0])
            a2 = np.array([3.0/2.0,-np.sqrt(3)/2.0])
        elif self.type == "triangle" or self.type=="kagome":
            a1 = np.array([1.0,0])
            a2 = np.array([0.5, np.sqrt(3)/2.0])
        elif self.type == "square":
            a1 = np.array([1.0,0])
            a2 = np.array([0,1.0])
        else: raise Exception(self.type + " is not implemented yet")
        return a1, a2

    def unit_cell(self):
        a1, a2 = self.translation_vecs()
        if self.type == "honeycomb":
            u1 = [0,0]
            u2 = [1.0, 0.0]
            s = [u1,u2]
        elif (self.type=="square" or self.type=="triangle"): 
            u1 = [0,0]
            s = [u1]
        elif self.type=="kagome":
            u1 = [0,0]
            u2 = 0.5*a1
            u3 = 0.5*a2
            s = [u1,u2,u3]
        return s

    def lattice_coords(self):
        coords = dict()
        a1, a2 = self.translation_vecs()
        uc_sites = self.unit_cell() 
        label = 1
        for n in range(self.lx):
            for m in range(self.ly):
                sites = uc_sites + n*a1 + m*a2
                for i in range(len(uc_sites)):
                    coords[label] = sites[i]
                    label += 1 
        return coords

class NN_bonds(lattice):
    def __init__(self, Type, Lx, Ly, boundary_condn:str):
        self.bc = boundary_condn
        super().__init__(Type, Lx, Ly)
    
    def distance(self,x,y):
        d = (x[0]-y[0])**2 + (x[1]-y[1])**2
        return np.sqrt(d)
    
    def bonds(self): 
        tol = 0.001
        lat_const = 0
        bond = []
        if self.type == "kagome": lat_const = 0.5
        else: lat_const = 1.0
        coords = self.lattice_coords()

        for i in range(1,len(coords)+1):
            for j in range(i+1,len(coords)+1):
                d = self.distance(coords[i], coords[j])
                if abs(d-lat_const) < tol:
                    bond.append([i , j])

        if (self.bc=="yperiodic" or self.bc=="periodic"):
            a1, a2 = self.translation_vecs()
            uc_atm = len(self.unit_cell())
            for k in range(1,len(coords)+1):
                if k%(uc_atm*self.ly) == 0:
                    edge = coords[k] - self.ly*a2
                    for l in range(1,k):
                        d = self.distance(edge,coords[l])
                        if abs(d-lat_const) < tol:
                            bond.append([k,l])
                            #print(k, l)

            if self.bc == "periodic":
                lim = len(coords) - self.ly*uc_atm
                for m in range(lim,len(coords)+1):
                    edge = coords[m] - self.lx*a1
                    for n in range(1,self.ly*uc_atm+1):
                        d = self.distance(edge,coords[n])
                        if abs(d-lat_const) < tol:
                            bond.append([m,n])
                
                topRsite = self.ly*uc_atm
                topRcoord = coords[topRsite]
                topRcoord = topRcoord - self.ly*a2 + self.lx*a1
                look = self.ly*uc_atm*(self.lx-1)
                for k in range(look, len(coords)):
                    d = self.distance(topRcoord,coords[k])
                    if abs(d-lat_const) < tol:
                        bond.append([topRsite,k])

        return bond
