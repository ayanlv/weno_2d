import matplotlib.cm as cm
import taichi as ti
import numpy as np

ti.init(arch = ti.gpu, default_fp = ti.f64)

# grid resolution
NX = 200
NY = 600
resx = 200
resy = 600

# size of grid
dx = 0.5 / NX
dy = 1.5 / NY

# time
dt = ti.field(ti.f64, shape=())
cfl = 0.5

cmap_name = 'jet'  # python colormap
img_field = 0      # 0:density, 1: schlieren, 2:vorticity, 3: velocity mag

# coordinate
coord = ti.Vector.field(2, ti.f64, shape=(NX + 6, NY + 6))

# conserved quantitiy
q = ti.Vector.field(4, ti.f64, shape=(NX + 6, NY + 6))
q_old = ti.Vector.field(4, ti.f64, shape=(NX + 6, NY + 6))
img = ti.field(ti.f64, shape=(resx + 6, resy + 6))

# flux
F = ti.Vector.field(4, ti.f64, shape=(NX + 6, NY + 6))
G = ti.Vector.field(4, ti.f64, shape=(NX + 6, NY + 6))
S = ti.Vector.field(4, ti.f64, shape=(NX + 6, NY + 6))
R = ti.Vector.field(4, ti.f64, shape=(NX + 6, NY + 6))

gam = 1.4  # ratio of specific heats

def initialize():
    for i in range(NX + 6):
        for j in range(NY + 6):
            coord[i, j][0] = (i - 2.5) * dx
            coord[i, j][1] = (j - 2.5) * dy - 0.75

    for i in range(NX + 6):
        for j in range(NY + 6):
            lamd = 0.01
            if coord[i, j][1] > 0.0:
                q[i, j][0] = 2.0
                q[i, j][1] = 0.0
                q[i, j][2] = 2.0 * lamd / 4 * (1 + np.cos(2 * np.pi * coord[i, j][0] / 0.5)) * (1 + np.cos(2 * np.pi * coord[i, j][1] / 1.5))
                q[i, j][3] = (2.5 - q[i, j][0] * coord[i, j][1]) / (gam - 1.0) + 0.5 * (q[i, j][1] ** 2 + q[i, j][2] ** 2) / q[i, j][0]
            else:
                q[i, j][0] = 1.0
                q[i, j][1] = 0.0
                q[i, j][2] = lamd / 4 * (1 + np.cos(2 * np.pi * coord[i, j][0] / 0.5)) * (1 + np.cos(2 * np.pi * coord[i, j][1] / 1.5))
                q[i, j][3] = (2.5 - q[i, j][0] * coord[i, j][1]) / (gam - 1.0) + 0.5 * (q[i, j][1] ** 2 + q[i, j][2] ** 2) / q[i, j][0]

@ti.kernel
def boundary():
    # Periodic Boundary
    for i, j in ti.ndrange(3, NY + 6):                 
        for k in ti.static(range(4)):
            q[i, j][k] = q[NX + i, j][k]    

    # Periodic Boundary
    for i, j in ti.ndrange((NX + 3, NX + 6), NY + 6):  
        for k in ti.static(range(4)):
            q[i, j][k] = q[i - NX, j][k]    

    # slip wall
    for i, j in ti.ndrange(NX + 6, 3):                  
        q[i, j][0] = q[i, 5 - j][0]
        q[i, j][1] = q[i, 5 - j][1]
        q[i, j][2] =-q[i, 5 - j][2]
        q[i, j][3] = q[i, 5 - j][3]

    # slip wall
    for i, j in ti.ndrange(NX + 6, (NY + 3, NY + 6)): 
        q[i, j][0] = q[i, 2 * NY + 5 - j][0]
        q[i, j][1] = q[i, 2 * NY + 5 - j][1]
        q[i, j][2] =-q[i, 2 * NY + 5 - j][2]
        q[i, j][3] = q[i, 2 * NY + 5 - j][3]

@ti.kernel
def get_dt():
    # timestep
    dt[None] = 1.0e5
    for i, j in q:
        prim = get_prim(q[i, j])
        a = ti.sqrt(gam * prim[3] / prim[0])
        dtij = cfl /((abs(prim[1]) + a) / dx + (abs(prim[2]) + a) / dy)
        ti.atomic_min(dt[None], dtij)

@ti.func
def get_prim(q):
    # convert conserved variables to primitive variables
    prim = ti.Vector([0.0, 0.0, 0.0, 0.0])
    prim[0] = q[0]  # rho
    prim[1] = q[1] / q[0]  # u
    prim[2] = q[2] / q[0]  # v
    prim[3] = (gam - 1) * (q[3] - 0.5 * (q[1] ** 2 + q[2] ** 2) / q[0])  # p
    return prim

@ti.kernel
def get_q_old():
    for i, j in q:
        q_old[i, j] = q[i, j]

@ti.func
def Rho_solver(qL, qR, n):
    # normal vector
    nx = n[0]
    ny = n[1]

    # Left state
    rL = qL[0]  # rho
    uL = qL[1] / qL[0]  # u
    vL = qL[2] / qL[0]  # v
    pL = (gam - 1.0) * (qL[3] - 0.5 * (qL[1] ** 2 + qL[2] ** 2) / qL[0])  #p
    
    vnL = uL * nx + vL * ny
    HL = (qL[3] + pL) / rL

    # Right state
    rR = qR[0]  # rho
    uR = qR[1] / qR[0]  # u
    vR = qR[2] / qR[0]  # v
    pR = (gam - 1.0) * (qR[3] - 0.5 * (qR[1] ** 2 + qR[2] ** 2) / qR[0])  #p
    vnR = uR * nx + vR * ny
    HR = (qR[3] + pR) / rR

    # Left and Right fluxes
    fL = ti.Vector([rL * vnL, rL * vnL * uL + pL * nx, rL * vnL * vL + pL * ny, rL * vnL * HL])
    fR = ti.Vector([rR * vnR, rR * vnR * uR + pR * nx, rR * vnR * vR + pR * ny, rR * vnR * HR])

    # Roe Averages
    rt = ti.sqrt(rR / rL)
    u = (uL + rt * uR) / (1.0 + rt)
    v = (vL + rt * vR) / (1.0 + rt)
    H = (HL + rt * HR) / (1.0 + rt)
    a = ti.sqrt((gam - 1.0) * (H - (u ** 2 + v ** 2) / 2.0))
    vn = u * nx + v * ny

    # eigenvalues
    kappa = 0.1
    epsilon = 2 * kappa * a
    lamd = ti.Vector([abs(vn), abs(vn + a), abs(vn - a)]) 
    for i in ti.static(range(3)):
        if lamd[i] < epsilon:
            lamd[i] = (lamd[i] ** 2 + epsilon ** 2) / 2.0 / epsilon

    # flux difference
    df = ti.Vector([2 * (a + abs(vn)) * (qR[0] - qL[0]), 2 * (a + abs(vn)) * (qR[1] - qL[1]), 2 * (a + abs(vn)) * (qR[2] - qL[2]), 2 * (a + abs(vn)) * (qR[3] - qL[3])])
    
    # Rho flux
    Rho_flux = ti.Vector([0.5 * (fL[0] + fR[0] - df[0]), 0.5 * (fL[1] + fR[1] - df[1]),
                          0.5 * (fL[2] + fR[2] - df[2]), 0.5 * (fL[3] + fR[3] - df[3])])

    return Rho_flux

@ti.func
def weno(q1, q2, q3, q4, q5, rl):
    # weno reconstruction
    q = ti.Vector([0.0, 0.0, 0.0, 0.0])
    for m in ti.static(range(4)):
        epsilon = 1.0e-20
        p = 2
        beta0 = 13 / 12 * (q1[m] - 2 * q2[m] + q3[m]) ** 2 + (q1[m] - 4 * q2[m] + 3 * q3[m]) ** 2 / 4
        beta1 = 13 / 12 * (q2[m] - 2 * q3[m] + q4[m]) ** 2 + (q2[m] - q4[m]) ** 2 / 4
        beta2 = 13 / 12 * (q3[m] - 2 * q4[m] + q5[m]) ** 2 + (3 * q3[m] - 4 * q4[m] + q5[m]) ** 2 / 4
        alpha0 = (1 + (abs(beta2 - beta0) / (beta0 + epsilon)) ** p) / 10
        alpha1 = (1 + (abs(beta2 - beta0) / (beta1 + epsilon)) ** p) * 3 / 5
        alpha2 = (1 + (abs(beta2 - beta0) / (beta2 + epsilon)) ** p) * 3 / 10
        w0 = alpha0 / (alpha0 + alpha1 + alpha2)
        w1 = alpha1 / (alpha0 + alpha1 + alpha2)
        w2 = alpha2 / (alpha0 + alpha1 + alpha2)
        if rl == 0: # L
            q[m] = w0 * (q1[m] / 3 - q2[m] * 7 / 6 + 11 * q3[m] / 6) + w1 * (-q2[m] / 6 + 5 * q3[m] / 6 + q4[m] / 3) + w2 * (q3[m] / 3 + 5 * q4[m] / 6 - q5[m] / 6)
        elif rl == 1: # R
            q[m] = w0 * (q1[m] / 3 - q2[m] * 7 / 6 + 11 * q3[m] / 6) + w1 * (-q2[m] / 6 + 5 * q3[m] / 6 + q4[m] / 3) + w2 * (q3[m] / 3 + 5 * q4[m] / 6 - q5[m] / 6)
    return q

@ti.kernel
def flux():
    for i, j in ti.ndrange((3, NX + 4), (3, NY + 3)):
        qL = ti.Vector([0.0, 0.0, 0.0, 0.0])
        qR = ti.Vector([0.0, 0.0, 0.0, 0.0])
        qL = weno(q[i - 3, j], q[i - 2, j], q[i - 1, j], q[i, j], q[i + 1, j], 0)
        qR = weno(q[i + 2, j], q[i + 1, j], q[i, j], q[i - 1, j], q[i - 2, j], 1) 
        F[i, j] = Rho_solver(qL, qR, ti.Vector([1.0, 0.0]))

    for i, j in ti.ndrange((3, NX + 3), (3, NY + 4)):
        qL = ti.Vector([0.0, 0.0, 0.0, 0.0])
        qR = ti.Vector([0.0, 0.0, 0.0, 0.0])
        qL = weno(q[i, j - 3], q[i, j - 2], q[i, j - 1], q[i, j], q[i, j + 1], 0)
        qR = weno(q[i, j + 2], q[i, j + 1], q[i, j], q[i, j - 1], q[i, j - 2], 1) 
        G[i, j] = Rho_solver(qL, qR, ti.Vector([0.0, 1.0]))
    
    for i, j in ti.ndrange((3, NX + 3), (3, NY + 3)):
        S[i, j] = [0, 0, -q[i, j][0], -q[i, j][2]]
        R[i, j] = (F[i, j] - F[i + 1, j]) / dx + (G[i, j] - G[i, j + 1]) / dy + S[i, j] 

@ti.kernel
def update(rk: ti.i32):
    for i, j in ti.ndrange((3, NX + 3), (3, NY + 3)):
        if rk == 0:
            q[i, j] = q_old[i, j] + dt[None] * R[i, j]
        if rk == 1:
            q[i, j] = (q[i, j] + 3 * q_old[i, j] + dt[None] * R[i, j]) / 4
        if rk == 2:
            q[i, j] = (2 * q[i, j] + q_old[i, j] + 2 * dt[None] * R[i, j]) / 3

@ti.kernel
def paint():
    for i, j in img:
        ii = i # min(max(1, i * NX // resx), NX - 2)
        jj = j # min(max(1, j * NY // resy), NY - 2)
        if img_field == 0:    # density
            img[i, j] = q[ii, jj][0]
        elif img_field == 1:  # numerical schlieren
            img[i,j] = ti.sqrt(((q[ii + 1, jj][0] - q[ii - 1, jj][0]) / dx)**2 +
                             ((q[ii, jj + 1][0] - q[ii, jj - 1][0]) / dy)**2)
        elif img_field == 2:  # vorticity
            img[i, j] = (q[ii + 1, jj][2] - q[ii - 1, jj][2]) / dx - (
                q[ii, jj + 1][1] - q[ii, jj - 1][1]) / dy
        elif img_field == 3:  # velocity magnitude
            img[i, j] = ti.sqrt(q[ii, jj][1]**2 + q[ii, jj][2]**2)

    #max = -1.0e10
    #min = 1.0e10
    #for i, j in img:
    #    ti.atomic_max(max, img[i, j])
    #    ti.atomic_min(min, img[i, j])
    max = 2.0
    min = 1.0
    for i, j in img:
        img[i, j] = (img[i, j] - min) / (max - min)


gui = ti.GUI('Euler Equations', (resx + 6, resy + 6))
cmap = cm.get_cmap(cmap_name)

n = 0
initialize()
boundary()
while gui.running:
    get_dt()
    get_q_old()
    for rk in range(3):
        flux()
        update(rk)
        boundary()

    if n == 53400:
        exit()
    if n % 100 == 0:
        paint()
        gui.set_image(cmap(img.to_numpy()))
        m = int(n/100)
        #gui.show("rh_new{:03d}.png".format(m))
        gui.show()
    n += 1
