
class finitedifference:
    
    def __init__(self):
        self
        #self.stepx = lambda

    def differentiate(self, g="NaN", a="NaN", b="NaN"):
        if type(g) == str:
            self.g = lambda x: 0
        else:
            self.g = g
        if type(a) == str:
            self.a = lambda t: 0
        else:
            self.a = a
        if type(b) == str:
            self.b = lambda t: 0
        else:
            self.b = b
    
        #boundary conditions
        u[:, 0] = self.g(x) # x \in [0, L]
        u[0, :] = self.a(t) # t >= 0
        u[-1, :] = self.b(t) # t >= 0

        # constant relation dx, dt
        alpha = dt/dx**2
        if alpha >= 0.5:
            raise valuerror:
                print("dt/dx^2 > 0.5. ensure relationship at or below 0.5")

        #differentiation loop
        for j in range(1, nT):
            for i in range(1, nX):
                u[i, j+1] =   alpha*u[i-1, j]\
                            + (1-2*alpha)*u[i,j]\
                            + alpha*u[i+1, j]
        
        #matrix multipl.
        #def. matrix A
        # Look up creating sparse matrices in python
        A = np.zeros[nX, nT]
        A[0, 0] = 1 - 2*alpha
        A[1, 0] = alpha
        A[0, 1] = alpha
        for i in range(1, nX-1):
            A[i,i] = 1-2*alpha
            A[i, i+1] = alpha
            A[i+1, i] = alpha
        #look up python vector matrix multiplications. specifically irt the @ operator
        V[0] = self.g(x)
        for j in range(nT):
            V[j+1] = A@V[j]

