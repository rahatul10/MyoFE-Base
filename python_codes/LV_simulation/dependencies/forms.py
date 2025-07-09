# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:06:26 2019
@author: ani228
"""

from dolfin import *
import sys
import numpy as np
from ufl import max_value, min_value, sign

class Forms(object):

    def __init__(self, params):    # amir: sth like constructor

        self.parameters = self.default_parameters()
        self.parameters.update(params)
        if "Fg" in self.parameters:
            self.Fg = self.parameters["Fg"]
            self.M1ij = self.parameters["M1ij"]
            self.M2ij = self.parameters["M2ij"]
            self.M3ij = self.parameters["M3ij"]
            #self.TF = self.parameters["TF"]
        else:
            self.Fg = Identity(3)

    def default_parameters(self):
        return {#"bff"  : 29.0,
			#"bfx"  : 13.3,
			#"bxx"  : 26.6,
			"Kappa": 1e5,
			"incompressible" : True,
			};


    def Fmat(self):

        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = Identity(d)
        F = I + grad(u)
        return F

    def update_Fg(self,theta1,theta2,theta3):
        Fg = self.Fg
        M1ij = self.M1ij
        M2ij = self.M2ij
        M3ij = self.M3ij
        #Fg = theta1*M1ij + theta2*M2ij + theta3*M3ij
        temp_Fg = theta1*M1ij + theta2*M2ij + theta3*M3ij
        Fg = project(temp_Fg,self.parameters["growth_tensor_FS"],
                    form_compiler_parameters={"representation":"uflacs"})
        #print "Fg updated", project(Fg,self.TF).vector().get_local()
        self.Fg = Fg

    def Fe(self):
        #Fg = self.parameters["growth_tensor"]
        F = self.Fmat()

        #Fg = self.Fg
        #Fe = as_tensor(F[i,j]*inv(Fg)[j,k], (i,k))
        if "Fg" in self.parameters:
            Fg = self.Fg
            #Fe = F* inv(Fg)
            Fe = as_tensor(F[i,j]*inv(Fg)[j,k], (i,k))
        else:
            Fe = F
        return Fe

    def Emat(self):

        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = Identity(d)
        #F = self.Fmat()
        F = self.Fe()
        #return 0.5*(F.T*F-I)
    	return 0.5*(as_tensor(F[k,i]*F[k,j] - I[i,j], (i,j)))


    def Cmat(self):

        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        #F = self.Fmat()
        F = self.Fe()
        return F.T*F
        #return as_tensor(F[k,i]*F[k,j],(i,j))

    def J(self):
        #F = self.Fmat()
        F = self.Fe()
        return det(F)


    def LVcavityvol(self):

        u = self.parameters["displacement_variable"]
        N = self.parameters["facet_normal"]
        mesh = self.parameters["mesh"]
        X = SpatialCoordinate(mesh)
        ds = dolfin.ds(subdomain_data = self.parameters["facetboundaries"])

        F = self.Fmat()
        #F = self.Fe()

        vol_form = -Constant(1.0/3.0) * inner(det(F)*dot(inv(F).T, N), X + u)*ds(self.parameters["LVendoid"])

        return assemble(vol_form, form_compiler_parameters={"representation":"uflacs"})
    
    def LVV0constrainedE(self):


        mesh = self.parameters["mesh"]
        u = self.parameters["displacement_variable"]
        ds = dolfin.ds(subdomain_data = self.parameters["facetboundaries"])
        dsendo = ds(self.parameters["LVendoid"], domain = self.parameters["mesh"], subdomain_data = self.parameters["facetboundaries"])
        pendo = self.parameters["lv_volconst_variable"]
        V0= self.parameters["lv_constrained_vol"]

        X = SpatialCoordinate(mesh)
        x = u + X

        F = self.Fmat()
        #F = self.Fe()

        N = self.parameters["facet_normal"]
        n = cofac(F)*N

        #n = det(F)*dot(inv(F).T, N)
        #vol_form = -Constant(1.0/3.0) * inner(det(F)*dot(inv(F).T, N), X + u)*ds(self.parameters["LVendoid"])

        area = assemble(Constant(1.0) * dsendo, form_compiler_parameters={"representation":"uflacs"})
        V_u = - Constant(1.0/3.0) * inner(n, x)
        #Wvol = (Constant(1.0/area) * pendo  * V0 * dsendo) - (pendo * V_u *dsendo)
        Wvol = (Constant(1.0/area) * pendo  * V0 * ds(self.parameters["LVendoid"])) - (pendo * V_u *ds(self.parameters["LVendoid"]))

        return Wvol

    def RVcavityvol(self):

        u = self.parameters["displacement_variable"]
        N = self.parameters["facet_normal"]
        mesh = self.parameters["mesh"]
        X = SpatialCoordinate(mesh)
        ds = dolfin.ds(subdomain_data = self.parameters["facetboundaries"])

        F = self.Fmat()

        vol_form = -Constant(1.0/3.0) * inner(det(F)*dot(inv(F).T, N), X + u)*ds(self.parameters["RVendoid"])

        return assemble(vol_form, form_compiler_parameters={"representation":"uflacs"})


    def LVcavitypressure(self):

        W = self.parameters["mixedfunctionspace"]
        w = self.parameters["mixedfunction"]
        mesh = self.parameters["mesh"]

        comm = W.mesh().mpi_comm()
        dofmap =  W.sub(self.parameters["LVendo_comp"]).dofmap()
        val_dof = dofmap.cell_dofs(0)[0]

	    # the owner of the dof broadcasts the value
        own_range = dofmap.ownership_range()

        try:
            val_local = w.vector()[val_dof][0]
        except IndexError:
                val_local = 0.0


        pressure = MPI.sum(comm, val_local)

        return pressure



    def RVcavitypressure(self):

        W = self.parameters["mixedfunctionspace"]
        w = self.parameters["mixedfunction"]
        mesh = self.parameters["mesh"]

        comm = W.mesh().mpi_comm()
        dofmap =  W.sub(self.parameters["RVendo_comp"]).dofmap()
        val_dof = dofmap.cell_dofs(0)[0]

	    # the owner of the dof broadcasts the value
        own_range = dofmap.ownership_range()

        try:
            val_local = w.vector()[val_dof][0]
        except IndexError:
            val_local = 0.0


        pressure = MPI.sum(comm, val_local)

        return pressure

    def TempActiveStress(self,time):

        f0 = self.parameters["fiber"]
        #cbforce = Expression('A*(B+sin((B/C)*time + D))', A=30000., B=1., C=16., D=80.2, time = time, degree=0)
        cbforce = Expression(("f"), f=0, degree=1)
        Pactive = cbforce * as_tensor(f0[i]*f0[j], (i,j))
        return Pactive, cbforce

#SARA: ADDING HO PASSIVE LAW
    def PassiveMatSEF(self,hsl):

        Ea = self.Emat()
        f0 = self.parameters["fiber"]
        s0 = self.parameters["sheet"]
        n0 = self.parameters["sheet-normal"]
        Kappa = self.parameters["Kappa"]
        passive_law = self.parameters["passive_law"][0]
        isincomp = self.parameters["incompressible"]
        Cmat = self.Cmat()
        hsl0 = self.parameters["hsl0"]
        

        if passive_law == "Guccione":

            C = self.parameters["c"][-1]
            C2 = self.parameters["c2"][-1]
            C3 = self.parameters["c3"][-1]
            bff = self.parameters["bf"][-1]
            bfx = self.parameters["bt"][-1]
            bxx = self.parameters["bfs"][-1]

            Eff = inner(f0, Ea*f0)
            Ess = inner(s0, Ea*s0)
            Enn = inner(n0, Ea*n0)
            Efs = inner(f0, Ea*s0)
            Efn = inner(f0, Ea*n0)
            Ens = inner(n0, Ea*s0)
            Esf = inner(s0, Ea*f0)
            Esn = inner(s0, Ea*n0)
            Enf = inner(n0, Ea*f0)

        
            if(isincomp):
                p = self.parameters["pressure_variable"]
                
            myofiber_stretch = hsl/hsl0
            QQ_m = conditional(myofiber_stretch > 1.0, C3*(myofiber_stretch - 1.0)**2.0, 0.0)

            #QQ_c = bff*Eff**2.0 + bfx*(Ess**2.0 + Enn**2.0 + 2.0*Ens**2.0) + bxx*(2.0*Efs**2.0 + 2.0*Efn**2.0)
            Qbulk = bff*Eff**2.0 + bfx*(Ess**2.0 + Enn**2.0 + Ens**2.0 + Esn**2.0) + bxx*(Efs**2.0 + Esf**2.0 + Efn**2.0 + Enf**2.0)
            #QQ_i = (C/2)*Eff**2 + bfx*(Ess**2.0 + Enn**2.0 + 2.0*Ens**2.0) + bxx*(2.0*Efs**2.0 + 2.0*Efn**2.0)


            Wp_m = C2*(exp(QQ_m) -  1.0)

            #Wp_m_weighted = phi_m*Wp_m
            Wp_m_weighted = Wp_m

            if(isincomp):
                Wp_c = C/2.0*(exp(Qbulk) -  1.0) - p*(self.J() - 1.0)
            else:
                Wp_c = C/2.0*(exp(QQ_c) -  1.0) + Kappa/2.0*(self.J() - 1.0)**2.0

            Wp = Wp_m + Wp_c
            #Wp = Wp_m_weighted + Wp_c_weighted
            #Wp = Wp_c
            return Wp
        elif passive_law == "Holzapfel":
            Ea = self.Emat()
            f0 = self.parameters["fiber"]
            s0 = self.parameters["sheet"]
            n0 = self.parameters["sheet-normal"]
            Kappa = self.parameters["Kappa"]
            passive_law = self.parameters["passive_law"][0]
            isincomp = self.parameters["incompressible"]
            Cmat = self.Cmat()
            hsl0 = self.parameters["hsl0"]

            # Material parameters
            C2 = self.parameters["c2"][-1]
            C3 = self.parameters["c3"][-1]
            a = self.parameters["a"][-1]
            b = self.parameters["hb"][-1]
            af = self.parameters["af"][-1]
            as_ = self.parameters["as_"][-1]
            bs = self.parameters["hbs"][-1]
            bf = self.parameters["hbf"][-1]
            bfs = self.parameters["hbfs"][-1]
            afs = self.parameters["afs"][-1]

            # Safety clamp for hsl0
            hsl0_safe = conditional(hsl0 < 1e-8, 1e-8, hsl0)
            myofiber_stretch = hsl / hsl0_safe

            # Invariants
            I1 = dolfin.variable(tr(Cmat))
            I4f = dolfin.variable(inner(f0, Cmat * f0))
            I4s = dolfin.variable(inner(s0, Cmat * s0))
            I8fs = dolfin.variable(inner(f0, Cmat * s0))

            # Clamp invariants to avoid overflow
            maxval = 100.0
            I1 = conditional(I1 > maxval, maxval, I1)
            I4f = conditional(I4f > maxval, maxval, I4f)
            I4s = conditional(I4s > maxval, maxval, I4s)
            I8fs = conditional(abs(I8fs) > maxval, maxval * sign(I8fs), I8fs)

            # Exponential terms with overflow protection
            Qbulk = (a / (2.0 * b)) * (exp(b * (I1 - 3.0)) - 1.0)
            Qfiber_sheet = (
                (af / (2.0 * bf)) * (exp(bf * (I4f - 1.0) ** 2.0) - 1.0) +
                (as_ / (2.0 * bs)) * (exp(bs * (I4s - 1.0) ** 2.0) - 1.0)
            )
            Qcoupling = (afs / (2.0 * bfs)) * (exp(bfs * I8fs ** 2.0) - 1.0)

            QQ_m = conditional(myofiber_stretch > 1.0, C3 * (myofiber_stretch - 1.0) ** 2.0, 0.0)
            Wp_m = C2 * (exp(QQ_m) - 1.0)

            # Incompressibility pressure
            if isincomp:
                try:
                    p = self.parameters["pressure_variable"]
                except KeyError:
                    p = Constant(0.0)
                Wp_c = Qbulk + Qfiber_sheet + Qcoupling - p * (self.J() - 1.0)
            else:
                Wp_c = Qbulk + Qfiber_sheet + Qcoupling + Kappa / 2.0 * (self.J() - 1.0)**2.0

            Wp = Wp_m + Wp_c

            # Debug print
            print("Passive SEF Debug |", 
                "I1 =", float(assemble(I1 * dx)), 
                "I4f =", float(assemble(I4f * dx)), 
                "I4s =", float(assemble(I4s * dx)), 
                "I8fs =", float(assemble(I8fs * dx)), 
                "Wp =", float(assemble(Wp * dx)))

            return Wp
            """C2 = self.parameters["c2"][-1]
            C3 = self.parameters["c3"][-1]
            a = self.parameters["a"][-1]
            b = self.parameters["hb"][-1]
            af = self.parameters["af"][-1]
            as_ = self.parameters["as_"][-1]
            bs = self.parameters["hbs"][-1]
            bf = self.parameters["hbf"][-1]
            bfs = self.parameters["hbfs"][-1]
            afs = self.parameters["afs"][-1]

            I1 = tr(Cmat)
            I4f = inner(f0, Cmat * f0)
            I4s = inner(s0, Cmat * s0)
            I8fs = inner(f0, Cmat * s0)
            if(isincomp):
                p = self.parameters["pressure_variable"]

            #Qbulk = b * (I1 - 3)
            Qbulk = (a / (2.0 * b)) * (exp(b * (I1 - 3.0)) - 1.0)
            Qfiber_sheet = (af / (2.0 * bf)) * (exp(bf * (I4f - 1.0) ** 2.0) - 1.0) + (as_ / (2.0 * bs)) * (exp(bs * (I4s - 1.0) ** 2.0) - 1.0)
            Qcoupling = (afs / (2.0 * bfs)) * (exp(bfs * I8fs ** 2.0) - 1.0)
        
                
            myofiber_stretch = hsl/hsl0
            QQ_m = conditional(myofiber_stretch > 1.0, C3*(myofiber_stretch - 1.0)**2.0, 0.0)

            Wp_m = C2 * (exp(QQ_m) - 1.0)
        
            Wp_c = Qbulk + Qfiber_sheet + Qcoupling - p*(self.J() - 1.0)
                #Wp_c = (a / (2.0 * b)) * (exp(Qbulk)) + Qfiber_sheet + Qcoupling - p*(self.J() - 1.0)
                #Wp_c = ((a / (2.0 * b)) * (exp(Qbulk)-1)) + Qfiber_sheet + Qcoupling - p*(self.J() - 1.0)
     
            Wp = Wp_m + Wp_c
            return Wp"""

    def PassiveMatSEFComps(self,hsl):
        Ea = self.Emat()
        f0 = self.parameters["fiber"]
        s0 = self.parameters["sheet"]
        n0 = self.parameters["sheet-normal"]
        Kappa = self.parameters["Kappa"]
        passive_law = self.parameters["passive_law"][0]
        isincomp = self.parameters["incompressible"]
        Cmat = self.Cmat()
        hsl0 = self.parameters["hsl0"]
        

        if passive_law == "Guccione":

            C = self.parameters["c"][-1]
            C2 = self.parameters["c2"][-1]
            C3 = self.parameters["c3"][-1]
            bff = self.parameters["bf"][-1]
            bfx = self.parameters["bt"][-1]
            bxx = self.parameters["bfs"][-1]

            Eff = inner(f0, Ea*f0)
            Ess = inner(s0, Ea*s0)
            Enn = inner(n0, Ea*n0)
            Efs = inner(f0, Ea*s0)
            Efn = inner(f0, Ea*n0)
            Ens = inner(n0, Ea*s0)
            Esf = inner(s0, Ea*f0)
            Esn = inner(s0, Ea*n0)
            Enf = inner(n0, Ea*f0)
            

        
            if(isincomp):
                p = self.parameters["pressure_variable"]
                
            myofiber_stretch = hsl/hsl0
            QQ_m = conditional(myofiber_stretch > 1.0, C3*(myofiber_stretch - 1.0)**2.0, 0.0)

            #QQ_c = bff*Eff**2.0 + bfx*(Ess**2.0 + Enn**2.0 + 2.0*Ens**2.0) + bxx*(2.0*Efs**2.0 + 2.0*Efn**2.0)
            Qbulk = bff*Eff**2.0 + bfx*(Ess**2.0 + Enn**2.0 + Ens**2.0 + Esn**2.0) + bxx*(Efs**2.0 + Esf**2.0 + Efn**2.0 + Enf**2.0)
            #QQ_i = (C/2)*Eff**2 + bfx*(Ess**2.0 + Enn**2.0 + 2.0*Ens**2.0) + bxx*(2.0*Efs**2.0 + 2.0*Efn**2.0)


            Wp_m = C2*(exp(QQ_m) -  1.0)


            if(isincomp):
                Wp_c = C/2.0*(exp(Qbulk) -  1.0) - p*(self.J() - 1.0)
            else:
                Wp_c = C/2.0*(exp(QQ_c) -  1.0) + Kappa/2.0*(self.J() - 1.0)**2.0

            Wp = Wp_m + Wp_c
            return Wp
        elif passive_law == "Holzapfel":
            C2 = self.parameters["c2"][-1]
            C3 = self.parameters["c3"][-1]
            a = self.parameters["a"][-1]
            b = self.parameters["hb"][-1]
            af = self.parameters["af"][-1]
            as_ = self.parameters["as_"][-1]
            bs = self.parameters["hbs"][-1]
            bf = self.parameters["hbf"][-1]
            bfs = self.parameters["hbfs"][-1]
            afs = self.parameters["afs"][-1]

            I1 = tr(Cmat)
            I4f = inner(f0, Cmat * f0)
            I4s = inner(s0, Cmat * s0)
            I8fs = inner(f0, Cmat * s0)

            Qbulk = (a / (2.0 * b)) * (exp(b * (I1 - 3.0)) - 1.0)
            Qfiber_sheet = (af / (2.0 * bf)) * (exp(bf * (I4f - 1.0) ** 2.0) - 1.0) + (as_ / (2.0 * bs)) * (exp(bs * (I4s - 1.0) ** 2.0) - 1.0)
            Qcoupling = (afs / (2.0 * bfs)) * (exp(bfs * I8fs ** 2.0) - 1.0)
            if(isincomp):
                p = self.parameters["pressure_variable"]
                
            myofiber_stretch = hsl/hsl0
            QQ_m = conditional(myofiber_stretch > 1.0, C3*(myofiber_stretch - 1.0)**2.0, 0.0)

            Wp_m = C2 * (exp(QQ_m) - 1.0)
            
            Wp_c = Qbulk + Qfiber_sheet + Qcoupling - p*(self.J() - 1.0)
                #Wp_c = (a / (2.0 * b)) * (exp(Qbulk)) + Qfiber_sheet + Qcoupling - p*(self.J() - 1.0)
                #Wp_c = ((a / (2.0 * b)) * (exp(Qbulk)-1)) + Qfiber_sheet + Qcoupling - p*(self.J() - 1.0)
            Wp = Wp_m + Wp_c
            return Wp


    def RVV0constrainedE(self):


        mesh = self.parameters["mesh"]
        self.parameters["displacement_variable"]
        ds = dolfin.ds(subdomain_data = self.parameters["facetboundaries"])
        dsendo = ds(self.parameters["RVendoid"], domain = self.parameters["mesh"], subdomain_data = self.parameters["facetboundaries"])
        pendo = self.parameters["rv_volconst_variable"]
        V0= self.parameters["rv_constrained_vol"]

        X = SpatialCoordinate(mesh)
        x = u + X

        F = self.Fmat()
        N = self.parameters["facet_normal"]
        n = cofac(F)*N

        area = assemble(Constant(1.0) * dsendo, form_compiler_parameters={"representation":"uflacs"})
        V_u = - Constant(1.0/3.0) * inner(x, n)
        Wvol = (Constant(1.0/area) * pendo  * V0 * dsendo) - (pendo * V_u *dsendo)

        return Wvol


#SARA: ADDING HO PASSIVE LAW
    def stress(self,hsl):
        mesh = self.parameters["mesh"]
        e1 = Constant((1.0, 0.0, 0.0))
        e2 = Constant((0.0, 1.0, 0.0))
        e3 = Constant((0.0, 0.0, 1.0))
        passive_law = self.parameters["passive_law"][0]
        isincomp = self.parameters["incompressible"]
        f0 = self.parameters["fiber"]
        s0 = self.parameters["sheet"]
        n0 = self.parameters["sheet-normal"]
        hsl0 = self.parameters["hsl0"]
        
        # Incompressibility parameter
        if isincomp:
            p = self.parameters["pressure_variable"]

        # Displacement-based quantities
        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = Identity(d)
        F = self.Fe()
        J = self.J()
        Ea = self.Emat()
        Ea = dolfin.variable(Ea)
        i, j, k, l = indices(4)
        Ctensor = self.Cmat()
        Ctensor = dolfin.variable(Ctensor)
        
        

        # --- Constitutive Law Selection ---
        if passive_law == "Guccione":
            # Extract parameters
            C = self.parameters["c"][-1]
            C2 = self.parameters["c2"][-1]
            C3 = self.parameters["c3"][-1]
            bff = self.parameters["bf"][-1]
            bfx = self.parameters["bt"][-1]
            bxx = self.parameters["bfs"][-1]

            Eff = f0[i]*Ea[i,j]*f0[j]
            Eff = dolfin.variable(Eff)
            Ess = s0[i]*Ea[i,j]*s0[j]
            Ess = dolfin.variable(Ess)
            Enn = n0[i]*Ea[i,j]*n0[j]
            Enn = dolfin.variable(Enn)
            Efs = f0[i]*Ea[i,j]*s0[j]
            Efs = dolfin.variable(Efs)
            Esf = s0[i]*Ea[i,j]*f0[j]
            Esf = dolfin.variable(Esf)
            Efn = f0[i]*Ea[i,j]*n0[j]
            Enf = n0[i]*Ea[i,j]*f0[j]
            Enf = dolfin.variable(Enf)
            Efn = dolfin.variable(Efn)
            Ens = n0[i]*Ea[i,j]*s0[j]
            Esn = s0[i]*Ea[i,j]*n0[j]
            Esn = dolfin.variable(Esn)
            Ens = dolfin.variable(Ens)
            
            # Myofiber stretch
            myofiber_stretch = hsl / hsl0
            Q = C3 * conditional(myofiber_stretch > 1.0, (myofiber_stretch - 1.0) ** 2.0, 0.0)

            # Passive myofiber stress
            Sff = (2.0 / myofiber_stretch) * C2 * C3 * (conditional(myofiber_stretch > 1.0, myofiber_stretch, 1.0) - 1.0) * exp(Q)

            # PK2 myofiber stress in local fiber coordinates
            S_local = as_tensor([[Sff, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            # Transformation matrix
            TransMatrix = as_tensor(f0[i] * e1[j], (i, j)) + as_tensor(s0[i] * e2[j], (i, j)) + as_tensor(n0[i] * e3[j], (i, j))

            S_global = TransMatrix * S_local * TransMatrix.T

            # Guccione SEF
            Qbulk = bff * Eff**2.0 + bfx * (Ess**2.0 + Enn**2.0 + Ens**2.0 + Esn**2.0) + bxx * (Efs**2.0 + Esf**2.0 + Efn**2.0 + Enf**2.0)
            Wp_c = (C / 2.0) * (exp(Qbulk) - 1.0)

            # Compute PK2 stress tensor
            PK2_local = as_tensor([
                [dolfin.diff(Wp_c, Eff), dolfin.diff(Wp_c, Efs), dolfin.diff(Wp_c, Efn)],
                [dolfin.diff(Wp_c, Esf), dolfin.diff(Wp_c, Ess), dolfin.diff(Wp_c, Esn)],
                [dolfin.diff(Wp_c, Enf), dolfin.diff(Wp_c, Ens), dolfin.diff(Wp_c, Enn)]
            ])
            PK2_global = as_tensor(TransMatrix[i, k] * TransMatrix[j, l] * PK2_local[k, l], (i, j))

        elif passive_law == "Holzapfel":
            #print("Using Holzapfel model")
            mesh = self.parameters["mesh"]
            e1 = Constant((1.0, 0.0, 0.0))
            e2 = Constant((0.0, 1.0, 0.0))
            e3 = Constant((0.0, 0.0, 1.0))
            passive_law = self.parameters["passive_law"][0]
            isincomp = self.parameters["incompressible"]
            f0 = self.parameters["fiber"]
            s0 = self.parameters["sheet"]
            n0 = self.parameters["sheet-normal"]
            hsl0 = self.parameters["hsl0"]

            if isincomp:
                try:
                    p = self.parameters["pressure_variable"]
                except KeyError:
                    p = Constant(0.0)

            u = self.parameters["displacement_variable"]
            d = u.ufl_domain().geometric_dimension()
            I = Identity(d)
            F = self.Fe()
            J = self.J()
            Ea = dolfin.variable(self.Emat())
            i, j, k, l = indices(4)
            Ctensor = dolfin.variable(self.Cmat())

            
                # Parameters
            C2 = self.parameters["c2"][-1]
            C3 = self.parameters["c3"][-1]
            a = self.parameters["a"][-1]
            b = self.parameters["hb"][-1]
            af = self.parameters["af"][-1]
            as_ = self.parameters["as_"][-1]
            bs = self.parameters["hbs"][-1]
            bf = self.parameters["hbf"][-1]
            bfs = self.parameters["hbfs"][-1]
            afs = self.parameters["afs"][-1]

                # Safety hsl0
            hsl0_safe = conditional(hsl0 < 1e-8, 1e-8, hsl0)
            myofiber_stretch = hsl / hsl0_safe

                # Invariants with clamping
            I1 = Ctensor[i, j] * Identity(3)[i, j]
            I1 = dolfin.variable(I1)

            I4f = f0[i] * Ctensor[i, j] * f0[j]
            I4f = dolfin.variable(I4f)

            I4s = s0[i] * Ctensor[i, j] * s0[j]
            I4s = dolfin.variable(I4s)

            I8fs = f0[i] * Ctensor[i, j] * s0[j]
            I8fs = dolfin.variable(I8fs)

            maxval = 100.0
            I1 = conditional(I1 > maxval, maxval, I1)
            I4f = conditional(I4f > maxval, maxval, I4f)
            I4s = conditional(I4s > maxval, maxval, I4s)
            I8fs = conditional(abs(I8fs) > maxval, maxval * sign(I8fs), I8fs)

                # Myofiber stress (passive)
            Q = C3 * conditional(myofiber_stretch > 1.0, (myofiber_stretch - 1.0)**2.0, 0.0)
            Sff = (2.0 / myofiber_stretch) * C2 * C3 * (conditional(myofiber_stretch > 1.0, myofiber_stretch, 1.0) - 1.0) * exp(Q)

            S_local = as_tensor([[Sff, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            TransMatrix = as_tensor(f0[i] * e1[j], (i, j)) + \
                        as_tensor(s0[i] * e2[j], (i, j)) + \
                        as_tensor(n0[i] * e3[j], (i, j))
            S_global = TransMatrix * S_local * TransMatrix.T

                # SEF energy components
            Qbulk = (a / (2.0 * b)) * (exp(b * (I1 - 3.0)) - 1.0)
            Qfiber_sheet = (
                (af / (2.0 * bf)) * (exp(bf * (I4f - 1.0)**2.0) - 1.0) +
                (as_ / (2.0 * bs)) * (exp(bs * (I4s - 1.0)**2.0) - 1.0)
            )
            Qcoupling = (afs / (2.0 * bfs)) * (exp(bfs * I8fs**2.0) - 1.0)
            Wp_c = Qbulk + Qfiber_sheet + Qcoupling

                # Derivatives
            dWp_dI1 = dolfin.diff(Wp_c, I1)
            dWp_dI4f = dolfin.diff(Wp_c, I4f)
            dWp_dI4s = dolfin.diff(Wp_c, I4s)
            dWp_dI8fs = dolfin.diff(Wp_c, I8fs)
            

                # PK2 stress tensor
            PK2_local = as_tensor([
                [2 * (dWp_dI1 + dWp_dI4f), 2 * dWp_dI8fs, 0.0],
                [2 * dWp_dI8fs, 2 * (dWp_dI1 + dWp_dI4s), 0.0],
                [0.0, 0.0, 2 * dWp_dI1]
            ])
            PK2_global = as_tensor(TransMatrix[i, k] * TransMatrix[j, l] * PK2_local[k, l], (i, j))

                # Final stress
            stress_tensor = S_global + PK2_global - p * inv(Ctensor)

                # Debug prints
            print("Passive SEF Debug |", 
                "I1 =", float(assemble(I1 * dx)), 
                "I4f =", float(assemble(I4f * dx)), 
                "I4s =", float(assemble(I4s * dx)), 
                "I8fs =", float(assemble(I8fs * dx)))
            
            return stress_tensor, Sff, S_global, PK2_global, -p * inv(Ctensor), myofiber_stretch

    def passivestress(self,hsl):

        mesh = self.parameters["mesh"]
        e1 = Constant((1.0, 0.0, 0.0))
        e2 = Constant((0.0, 1.0, 0.0))
        e3 = Constant((0.0, 0.0, 1.0))
        passive_law = self.parameters["passive_law"][0]
        isincomp = self.parameters["incompressible"]
        f0 = self.parameters["fiber"]
        s0 = self.parameters["sheet"]
        n0 = self.parameters["sheet-normal"]
        hsl0 = self.parameters["hsl0"]
        
        # Incompressibility parameter
        if isincomp:
            p = self.parameters["pressure_variable"]

        # Displacement-based quantities
        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = Identity(d)
        F = self.Fe()
        J = self.J()
        Ea = self.Emat()
        Ea = dolfin.variable(Ea)
        i, j, k, l = indices(4)
        Ctensor = self.Cmat()
        Ctensor = dolfin.variable(Ctensor)
        # Ensure Ctensor is a differentiable UFL Variable

            # Compute strain components
        Eff = f0[i]*Ea[i,j]*f0[j]
        Eff = dolfin.variable(Eff)
        Ess = s0[i]*Ea[i,j]*s0[j]
        Ess = dolfin.variable(Ess)
        Enn = n0[i]*Ea[i,j]*n0[j]
        Enn = dolfin.variable(Enn)
        Efs = f0[i]*Ea[i,j]*s0[j]
        Efs = dolfin.variable(Efs)
        Esf = s0[i]*Ea[i,j]*f0[j]
        Esf = dolfin.variable(Esf)
        Efn = f0[i]*Ea[i,j]*n0[j]
        Enf = n0[i]*Ea[i,j]*f0[j]
        Enf = dolfin.variable(Enf)
        Efn = dolfin.variable(Efn)
        Ens = n0[i]*Ea[i,j]*s0[j]
        Esn = s0[i]*Ea[i,j]*n0[j]
        Esn = dolfin.variable(Esn)
        Ens = dolfin.variable(Ens)


        # --- Constitutive Law Selection ---
        if passive_law == "Guccione":
            # Extract parameters
            C = self.parameters["c"][-1]
            C2 = self.parameters["c2"][-1]
            C3 = self.parameters["c3"][-1]
            bff = self.parameters["bf"][-1]
            bfx = self.parameters["bt"][-1]
            bxx = self.parameters["bfs"][-1]
            # Myofiber stretch
            myofiber_stretch = hsl / hsl0
            Q = C3 * conditional(myofiber_stretch > 1.0, (myofiber_stretch - 1.0) ** 2.0, 0.0)

            # Passive myofiber stress
            Sff = (2.0 / myofiber_stretch) * C2 * C3 * (conditional(myofiber_stretch > 1.0, myofiber_stretch, 1.0) - 1.0) * exp(Q)

            # PK2 myofiber stress in local fiber coordinates
            S_local = as_tensor([[Sff, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            # Transformation matrix
            TransMatrix = as_tensor(f0[i] * e1[j], (i, j)) + as_tensor(s0[i] * e2[j], (i, j)) + as_tensor(n0[i] * e3[j], (i, j))

            S_global = TransMatrix * S_local * TransMatrix.T

            # Guccione SEF
            Qbulk = bff * Eff**2.0 + bfx * (Ess**2.0 + Enn**2.0 + Ens**2.0 + Esn**2.0) + bxx * (Efs**2.0 + Esf**2.0 + Efn**2.0 + Enf**2.0)
            Wp_c = (C / 2.0) * (exp(Qbulk) - 1.0)

            # Compute PK2 stress tensor
            PK2_local = as_tensor([
                [dolfin.diff(Wp_c, Eff), dolfin.diff(Wp_c, Efs), dolfin.diff(Wp_c, Efn)],
                [dolfin.diff(Wp_c, Esf), dolfin.diff(Wp_c, Ess), dolfin.diff(Wp_c, Esn)],
                [dolfin.diff(Wp_c, Enf), dolfin.diff(Wp_c, Ens), dolfin.diff(Wp_c, Enn)]
            ])
            PK2_global = as_tensor(TransMatrix[i, k] * TransMatrix[j, l] * PK2_local[k, l], (i, j))

        elif passive_law == "Holzapfel":
            C2 = self.parameters["c2"][-1]
            C3 = self.parameters["c3"][-1]
            a = self.parameters["a"][-1]
            b = self.parameters["hb"][-1]
            af = self.parameters["af"][-1]
            as_ = self.parameters["as_"][-1]
            bs = self.parameters["hbs"][-1]
            bf = self.parameters["hbf"][-1]
            bfs = self.parameters["hbfs"][-1]
            afs = self.parameters["afs"][-1]
            Eff = f0[i]*Ea[i,j]*f0[j]
            Eff = dolfin.variable(Eff)
            # Invariant calculations
            #I1 = tr(Ctensor)
            #I4f = inner(f0, Ctensor * f0)
            #I4s = inner(s0, Ctensor * s0)
            #I8fs = inner(f0, Ctensor * s0)
        
            I1 = Ctensor[i, j] * Identity(3)[i, j]
            I1 = dolfin.variable(I1)

            I4f = f0[i] * Ctensor[i, j] * f0[j]
            I4f = dolfin.variable(I4f)

            I4s = s0[i] * Ctensor[i, j] * s0[j]
            I4s = dolfin.variable(I4s)

            I8fs = f0[i] * Ctensor[i, j] * s0[j]
            I8fs = dolfin.variable(I8fs)

            myofiber_stretch = hsl / hsl0
            Q = C3 * conditional(myofiber_stretch > 1.0, (myofiber_stretch - 1.0) ** 2.0, 0.0)

            # Passive myofiber stress
            Sff = (2.0 / myofiber_stretch) * C2 * C3 * (conditional(myofiber_stretch > 1.0, myofiber_stretch, 1.0) - 1.0) * exp(Q)
            S_local = as_tensor([[Sff, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            # Transformation matrix
            TransMatrix = as_tensor(f0[i] * e1[j], (i, j)) + as_tensor(s0[i] * e2[j], (i, j)) + as_tensor(n0[i] * e3[j], (i, j))

            S_global = TransMatrix * S_local * TransMatrix.T
            # Holzapfel SEF
            Qbulk = (a / (2.0 * b)) * (exp(b * (I1 - 3.0)) - 1.0)
            Qfiber_sheet = (af / (2.0 * bf)) * (exp(bf * (I4f - 1.0) ** 2.0) - 1.0) + (as_ / (2.0 * bs)) * (exp(bs * (I4s - 1.0) ** 2.0) - 1.0)
            Qcoupling = (afs / (2.0 * bfs)) * (exp(bfs * I8fs ** 2.0) - 1.0)
            Wp_c = Qbulk + Qfiber_sheet + Qcoupling 
            #Wp_c = (a / (2.0 * b)) * (exp(Qbulk)) + Qfiber_sheet + Qcoupling 
            #Wp_c = ((a / (2.0 * b)) * (exp(Qbulk)-1)) + Qfiber_sheet + Qcoupling

            # Compute PK2 stress tensor
            # Compute derivatives of Wp_c w.r.t. invariants
        
            dWp_dI1 = dolfin.diff(Wp_c, I1)
            dWp_dI4f = dolfin.diff(Wp_c, I4f)
            dWp_dI4s = dolfin.diff(Wp_c, I4s)
            dWp_dI8fs = dolfin.diff(Wp_c, I8fs)

                # Compute derivatives of invariants w.r.t. Ctensor
            """dI1_dC = Identity(3)  
                dI1_dC = dolfin.variable(dI1_dC)
                dI4f_dC = f0[i] * f0[j]
                dI4f_dC = dolfin.variable(dI4f_dC)
                dI4s_dC = s0[i] * s0[j] 
                dI4s_dC = dolfin.variable(dI4s_dC)
                dI8fs_dC = f0[i] * s0[j] + f0[j] * s0[i] 
                dI8fs_dC = dolfin.variable(dI8fs_dC)"""

                # Assemble the PK2 stress tensor
            PK2_local = as_tensor([
                [2*(dWp_dI1 + dWp_dI4f), 2*(dWp_dI8fs), 0.0],
                [2*dWp_dI8fs, 2*(dWp_dI1 + dWp_dI4s), 0.0],
                [0.0, 0.0, 2*dWp_dI1]
            ])
            PK2_global = as_tensor(TransMatrix[i,k]*TransMatrix[j,l]*PK2_local[k,l],(i,j))

        else:
            raise ValueError("Unknown passive law: {}".format(passive_law))
        
        return PK2_local,-p*inv(Ctensor)

    def return_radial_vec_ratio(self):

        mesh = self.parameters["mesh"]
        s0 = self.parameters["sheet"]
        print s0[0]

        X = SpatialCoordinate(mesh)
        ratio = s0_evaluated.y()/s0_evaluated.x()

        return ratio

    def Umat(self):

        Fmat = self.Fmat()
        #Fmat = self.Fe()
        F0 = Fmat
        for j in range(15):
            F0 = 0.5* (F0 + inv(F0).T)
        R = F0
        return inv(R)*Fmat

    def kroon_law(self,FunctionSpace,step_size,kappa,binary_mask):

        mesh = self.parameters["mesh"]
        C = self.Cmat()
        f0 = self.parameters["fiber"]
        f = C*f0/sqrt(inner(C*f0,C*f0))
	f_proj = project(f,VectorFunctionSpace(mesh,"DG",1),form_compiler_parameters={"representation":"uflacs"})
	"""for i in range(len(binary_mask)):
            f_array = f_proj.vector().get_local()[i*3:(i+1)*3]
            if binary_mask[i] == 1:

                f_proj.vector()[i*3] = f0.vector().get_local()[i*3]
                f_proj.vector()[i*3+1] = f0.vector().get_local()[i*3+1]
                f_proj.vector()[i*3+2] = f0.vector().get_local()[i*3+2]"""  #SARA: RELATED TO FIBER-REMODELING
        f_adjusted = 1./kappa * (f_proj - f0) * step_size
        f_adjusted = project(f_adjusted,VectorFunctionSpace(mesh,"DG",1),form_compiler_parameters={"representation":"uflacs"})
        f_adjusted = interpolate(f_adjusted,FunctionSpace)

        return f_adjusted

    def eigen(self,T,dgs,dgv):

        mesh = self.parameters["mesh"]

        dofmap = dgs.dofmap()
        dofs = dofmap.dofs()

        eigval1 = Function(dgs)
        eigval2 = Function(dgs)
        eigval3 = Function(dgs)

        eigvec1 = Function(dgv)
        eigvec2 = Function(dgv)
        eigvec3 = Function(dgv)


        E1 = eigval1.vector().array()
        E2 = eigval2.vector().array()
        E3 = eigval3.vector().array()

        V1 = eigvec1.vector().array().reshape([len(dofs),3])
        V2 = eigvec2.vector().array().reshape([len(dofs),3])
        V3 = eigvec3.vector().array().reshape([len(dofs),3])


        Emax = E1
        Emin = E3
        E3rd = E2

        Vmax = V1
        Vmin = V2
        V3rd = V3

        print "calculating eigenvalue"

        F = T.vector().array()
        if all(np.equal(F,np.zeros(len(F)))) == True:

            return 'zero array'

        else:

            mesh1 = T.function_space().mesh()

            #print len(F)
            #print np.shape(F), len(dofs)
            #print mesh1

            gdim = mesh.geometry().dim()
            #print gdim

            # Get coordinates as len(dofs) x gdim array
            dofs_x = dgs.tabulate_dof_coordinates().reshape((-1, gdim))

            RC = F.reshape([len(dofs),3,3])

            RC = np.where(RC< 1e-10,0.,RC)

            for idx, (dof,x, v) in enumerate(zip(dofs, dofs_x, RC)):
                #print idx, dof, x, v
                [eigL,eigR] = np.linalg.eig(v)


                ls1 = eigL[0]
                ls2 = eigL[1]
                ls3 = eigL[2]


                lv1 = eigR[:,0]
                lv2 = eigR[:,1]
                lv3 = eigR[:,2]

                lv1 = lv1/np.dot(lv1, lv1)
                lv2 = lv2/np.dot(lv2, lv2)
                lv3 = lv3/np.dot(lv3, lv3)


                lsmax = max(ls1, ls2, ls3)

                lsmin = min(ls1, ls2, ls3)
                if lsmax == ls1:
                    Vmax[idx] = lv1
                    if lsmin==ls2:
                        ls3rd = ls3
                        Vmin[idx] = lv2
                        V3rd[idx] = lv3
                    elif lsmin == ls3:
                        ls3rd = ls2
                        Vmin[idx] = lv3
                        V3rd[idx] = lv2

                elif lsmax == ls2:
                    Vmax[idx] = lv2
                    if lsmin == ls1:
                        ls3rd = ls3
                        Vmin[idx] = lv1
                        V3rd[idx] = lv3
                    elif lsmin == ls3:
                        ls3rd = ls1
                        Vmin[idx] = lv3
                        V3rd[idx] = lv1

                elif lsmax == ls3:
                    Vmax[idx] = lv3
                    if lsmin == ls1:
                        ls3rd = ls2
                        Vmin[idx] = lv1
                        V3[idx] = lv2
                    elif lsmin == ls2:
                        ls3rd = ls1
                        Vmin[idx] = lv2
                        V3rd[idx] = lv1

                if Vmax[idx,0] < 0:
                    Vmax [idx,:] *= -1
                else:
                    pass

                Emax[idx] = lsmax
                Emin[idx] = lsmin
                E3rd[idx] = ls3rd

            maxvec = Function(dgv)
            maxvec.vector().set_local(Vmax.flatten())
            maxvec.vector().apply("insert")

            minvec = Function(dgv)
            minvec.vector().set_local(Vmin.flatten())
            minvec.vector().apply("insert")

            vec3rd = Function(dgv)
            vec3rd.vector().set_local(V3rd.flatten())
            vec3rd.vector().apply("insert")

            emax = Function(dgs)
            emin = Function(dgs)
            e3rd = Function(dgs)

            emax.vector().set_local(Emax.flatten())
            emax.vector().apply("insert")

            emin.vector().set_local(Emin.flatten())
            emin.vector().apply("insert")

            e3rd.vector().set_local(E3rd.flatten())
            e3rd.vector().apply("insert")


            return maxvec


    def stress_kroon(self,stress_tensor,FS,VFS,TFS,step_size,kappa):

        mesh = self.parameters["mesh"]
        f0 = self.parameters["fiber"]
        #inv_F = inv(self.Fmat())
        eigen = self.eigen(stress_tensor,FS,VFS)

        if eigen == 'zero array':
            f = f0
        else:
            #print "eigen"
            #print eigen.vector().get_local().reshape(24,3)[0:4]
            f = eigen
            f /= sqrt(inner(f,f))



        f_adjusted = 1./kappa * (f - f0) * step_size
        f_adjusted = project(f_adjusted,VectorFunctionSpace(mesh,"DG",1),form_compiler_parameters={"representation":"uflacs"})
        #print 'f_adj before interpolate: '
        #print f_adjusted.vector().get_local()[0:4]
        f_adjusted = interpolate(f_adjusted,VFS)
        #print 'f_adj after interpolate: '
        #print f_adjusted.vector().get_local()[0:4]

        return f_adjusted

    def new_stress_kroon(self,stress_tensor,FunctionSpace,step_size,kappa,binary_mask):

        mesh = self.parameters["mesh"]
        PK2 = stress_tensor
        f0 = self.parameters["fiber"]
        f = PK2*f0/sqrt(inner(PK2*f0,PK2*f0))

	f_proj = project(f,VectorFunctionSpace(mesh,"DG",1),form_compiler_parameters={"representation":"uflacs"})
        """for i in range(no_of_int_points):
            f_array = f_proj.vector().get_local()[i*3:(i+1)*3]
            if np.all(np.isnan(f_array)):
		f_proj.vector()[i*3] = f0.vector().get_local()[i*3]
                f_proj.vector()[i*3+1] = f0.vector().get_local()[i*3+1]
                f_proj.vector()[i*3+2] = f0.vector().get_local()[i*3+2]"""
	"""for i in range(len(binary_mask)):
            f_array = f_proj.vector().get_local()[i*3:(i+1)*3]
            if binary_mask[i] == 1:
                f_proj.vector()[i*3] = f0.vector().get_local()[i*3]
                f_proj.vector()[i*3+1] = f0.vector().get_local()[i*3+1]
                f_proj.vector()[i*3+2] = f0.vector().get_local()[i*3+2]"""

        """for index in np.arange(len(binary_mask)):
            if binary_mask[index] == 1:
                f.vector()[index*3] = f0.vector().get_local()[index*3]
                f.vector()[index*3+1] = f0.vector().get_local()[index*3+1]
                f.vector()[index*3+2] = f0.vector().get_local()[index*3+2]"""

        f_adjusted = 1./kappa * (f_proj - f0) * step_size
        f_adjusted = project(f_adjusted,VectorFunctionSpace(mesh,"DG",1),form_compiler_parameters={"representation":"uflacs"})
        f_adjusted = interpolate(f_adjusted,FunctionSpace)

        return f_adjusted


    def rand_walk(self,width):


        f0 = self.parameters["fiber"]
        for i in np.arange(np.shape(f0.vector().array())[0]/3):
            i = int(i)
            f0.vector()[i*3] = np.random.normal(f0.vector().array()[i*3],width)
            f0.vector()[i*3+1] = np.random.normal(f0.vector().array()[i*3+1],width)
            f0.vector()[i*3+2] = np.random.normal(f0.vector().array()[i*3+2],width)

        return f0
    
    def F_print(self,F):

        mesh = self.parameters["mesh"]
        
        #F = self.Fe()
        fs = TensorFunctionSpace(mesh, "DG", 0)
        fs._quad_scheme = 'default'
        fs_proj = project(F,fs,
                form_compiler_parameters={"representation":"uflacs"})

        F_projected = Function(fs)
        F_values = fs_proj.vector().get_local()

        # Get the mesh from F
        mesh = fs_proj.function_space().mesh()
        cell_dofs = fs_proj.function_space().dofmap().cell_dofs
        num_cells = mesh.num_cells()

        # Define the dimension of the deformation gradient tensor
        dim = 3  # Assuming 3D problem with 3x3 tensor


        cell_id = 10    
        dofs = cell_dofs(cell_id)
        # Extract the tensor values for this cell
        tensor_values = np.array(F_values[dofs]).reshape((dim, dim))

        return tensor_values

    def cycle_strain(self):

        mesh = self.parameters["mesh"]
        
        F = self.Fe()