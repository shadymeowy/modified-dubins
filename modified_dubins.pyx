import cython
from libc.math cimport sqrt, sin, cos, atan2, M_PI, isfinite

cpdef enum DUBINS_PATH_TYPE:
    RSR = 0x1
    RSL = 0x2
    LSL = 0x4
    LSR = 0x8
    RRR = 0x10
    LLL = 0x20
    CSC = RSR | RSL | LSL | LSR
    CCC = RRR | LLL

# An ugly hack to get around Cython generating duplicate array
# conversion code with the same function name...
ctypedef double double1
ctypedef double double2
ctypedef double double3

cdef:
    struct dubins_args:
        double[3] e_z
        double[3] position
        double[3] direction
        double[3] target_position
        double[3] target_direction
        double r
    
    struct dubins_path:
        DUBINS_PATH_TYPE type
        double1[3] n
        double r
        double cost
        double m_p
        double m_t
        double1[3] l1
        double1[3] l2
        double1[3] r_p
        double1[3] v_p
        double theta_p
        double1[3] r_t
        double1[3] v_t
        double theta_t
        double2[3] r_c
        double2[3] v_c
        double theta_c
        
    struct dubins_paths:
        int count
        dubins_path[6] paths

cdef inline void add_v(double[3] a, double[3] b, double[3] o) nogil:
    o[0] = a[0]+b[0]
    o[1] = a[1]+b[1]
    o[2] = a[2]+b[2]

cdef inline void sub_v(double[3] a, double[3] b, double[3] o) nogil:
    o[0] = a[0]-b[0]
    o[1] = a[1]-b[1]
    o[2] = a[2]-b[2]

cdef inline void mul_v(double[3] a, double b, double[3] o) nogil:
    o[0] = a[0]*b
    o[1] = a[1]*b
    o[2] = a[2]*b

@cython.cdivision(True)
cdef inline void div_v(double[3] a, double b, double[3] o) nogil:
    o[0] = a[0]/b
    o[1] = a[1]/b
    o[2] = a[2]/b

cdef inline double dot(double[3] a, double[3] b) nogil:
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

cdef inline double norm(double[3] v) nogil:
    return sqrt(dot(v, v))

cdef inline void normalize(double[3] v, double[3] o) nogil:
    cdef double l = norm(v)
    div_v(v, l, o)

cdef inline void negate_v(double[3] v, double[3] o) nogil:
    o[0] = -v[0]
    o[1] = -v[1]
    o[2] = -v[2]

cdef inline void cross(double[3] a, double[3] b, double[3] o) nogil:
    o[0] = a[1]*b[2]-a[2]*b[1]
    o[1] = a[2]*b[0]-a[0]*b[2]
    o[2] = a[0]*b[1]-a[1]*b[0]
    
cdef inline void cross3(double[3] v1, double[3] v2, double[3] o) nogil:
    cdef double[3] t
    # To be replaced with a simple expr
    cross(v1, v2, t)
    cross(v2, t, o)

cdef inline double angle(double[3] v1, double[3] v2, double[3] n) nogil:
    cdef double[3] t
    cross(v1, v2, t)
    # To be replaced with a branchless and approximated expr
    cdef double r = atan2(dot(n, t), dot(v1, v2))
    r += 2*M_PI if r < 0 else 0
    return r

@cython.cdivision(True)
cpdef dubins_paths dubins(dubins_args args):
    cdef:
        dubins_paths paths
        dubins_path path
        double[3] delta_pos
        double[3] n
        double[3] vp
        double[3] vt
        double[3] r_pl
        double[3] r_pr
        double[3] r_tl
        double[3] r_tr
        double[3] v
        double[3] t
        double L
        double[3] Lv
    
    normalize(args.direction, args.direction)
    normalize(args.target_direction, args.target_direction)
    sub_v(args.target_position, args.position, delta_pos)
    
    cross3(args.e_z, delta_pos, n)
    normalize(n, n)
    
    cross(n, args.direction, vp)
    normalize(vp, vp)
    cross(n, args.target_direction, vt)
    normalize(vt, vt)
    
    # TODO need a check for degenerate cases
    args.r /= dot(args.e_z, n)
    
    mul_v(vp, args.r, r_pl)
    add_v(args.position, r_pl, r_pl)
    mul_v(vp, args.r, r_pr)
    sub_v(args.position, r_pr, r_pr)
    
    mul_v(vt, args.r, r_tl)
    add_v(args.target_position, r_tl, r_tl)
    mul_v(vt, args.r, r_tr)
    sub_v(args.target_position, r_tr, r_tr)
    
    paths.count = 0
    path.n = n
    path.r = args.r
    
    # RSR
    path.type = DUBINS_PATH_TYPE.RSR
    path.v_p = vp
    path.r_p = r_pr
    path.r_t = r_tr

    sub_v(r_tr, r_pr, v)

    cross(n, v, path.v_t)
    normalize(path.v_t, path.v_t)

    path.theta_p = angle(path.v_t, vp, n)
    negate_v(n, n)
    path.theta_t = angle(path.v_t, vt, n)
    negate_v(n, n)

    mul_v(path.v_t, args.r, path.l1)
    add_v(r_pr, path.l1, path.l1)
    mul_v(path.v_t, args.r, path.l2)
    add_v(r_tr, path.l2, path.l2)

    path.cost = args.r * (abs(path.theta_p) + abs(path.theta_t)) + norm(v)
    
    if isfinite(path.cost):
        paths.paths[paths.count] = path
        paths.count += 1
    
    # RSL
    path.type = DUBINS_PATH_TYPE.RSL
    path.v_p = vp
    path.r_p = r_pr
    path.r_t = r_tl

    sub_v(r_tl, r_pr, v)

    L = sqrt(dot(v, v)-4*args.r*args.r)

    cross(n, v, path.v_t)
    mul_v(path.v_t, L, path.v_t)
    mul_v(v, 2*args.r, t)
    add_v(path.v_t, t, path.v_t)
    normalize(path.v_t, path.v_t)

    path.theta_p = angle(path.v_t, vp, n)
    path.theta_t = -angle(path.v_t, vt, n)

    mul_v(path.v_t, args.r, path.l1)
    add_v(r_pr, path.l1, path.l1)
    mul_v(path.v_t, -args.r, path.l2)
    add_v(r_tl, path.l2, path.l2)

    path.cost = args.r * (abs(path.theta_p)+abs(path.theta_t)) + L
    negate_v(path.v_t, path.v_t)
    
    if isfinite(path.cost):
        paths.paths[paths.count] = path
        paths.count += 1
    
    # LSL
    path.type = DUBINS_PATH_TYPE.LSL
    path.v_p = vp
    negate_v(path.v_p, path.v_p)
    path.r_p = r_pl
    path.r_t = r_tl

    sub_v(r_tl, r_pl, v)

    cross(v, n, path.v_t)
    normalize(path.v_t, path.v_t)

    negate_v(vp, vp)
    negate_v(n, n)
    path.theta_p = -angle(path.v_t, vp, n)
    negate_v(n, n)
    negate_v(vp, vp)
    negate_v(vt, vt)
    path.theta_t = -angle(path.v_t, vt, n)
    negate_v(vt, vt)

    mul_v(path.v_t, args.r, path.l1)
    add_v(r_pl, path.l1, path.l1)
    mul_v(path.v_t, args.r, path.l2)
    add_v(r_tl, path.l2, path.l2)

    path.cost = args.r * (abs(path.theta_p) + abs(path.theta_t)) + norm(v)
    
    if isfinite(path.cost):
        paths.paths[paths.count] = path
        paths.count += 1

    # LSR
    path.type = DUBINS_PATH_TYPE.LSR
    path.v_p = vp
    negate_v(path.v_p, path.v_p)
    path.r_p = r_pl
    path.r_t = r_tr

    sub_v(r_tr, r_pl, v)

    L = sqrt(dot(v, v)-4*args.r*args.r)

    cross(n, v, path.v_t)
    mul_v(path.v_t, -L, path.v_t)
    mul_v(v, 2*args.r, t)
    add_v(path.v_t, t, path.v_t)
    normalize(path.v_t, path.v_t)

    negate_v(n, n)
    negate_v(vp, vp)
    path.theta_p = -angle(path.v_t, vp, n)
    negate_v(vp, vp)
    negate_v(path.v_t, path.v_t)
    path.theta_t = angle(path.v_t, vt, n)
    negate_v(path.v_t, path.v_t)
    negate_v(n, n)

    mul_v(path.v_t, args.r, path.l1)
    add_v(r_pl, path.l1, path.l1)
    mul_v(path.v_t, -args.r, path.l2)
    add_v(r_tr, path.l2, path.l2)

    path.cost = args.r * (abs(path.theta_p)+abs(path.theta_t)) + L
    negate_v(path.v_t, path.v_t)
    
    if isfinite(path.cost):
        paths.paths[paths.count] = path
        paths.count += 1

    # RRR
    path.type = DUBINS_PATH_TYPE.RRR
    path.r_p = r_pr
    path.v_p = vp
    path.r_t = r_tr

    sub_v(r_tr, r_pr, v)
    div_v(v, 2, v)

    cross(v, n, Lv)
    normalize(Lv, Lv)
    mul_v(Lv, sqrt(4*args.r*args.r - dot(v, v)), Lv)

    sub_v(Lv, v, path.v_t)
    normalize(path.v_t, path.v_t)
    add_v(Lv, v, path.v_c)
    normalize(path.v_c, path.v_c)

    mul_v(path.v_c, 2*args.r, path.r_c)
    add_v(path.r_c, r_pr, path.r_c)

    path.theta_p = angle(path.v_c, vp, n)
    negate_v(n, n)
    path.theta_t = angle(path.v_t, vt, n)
    negate_v(n, n)
    path.theta_c = -angle(path.v_c, path.v_t, n)

    path.cost = args.r * (abs(path.theta_p)+abs(path.theta_t)+abs(path.theta_c))
    negate_v(path.v_c, path.v_c)
    
    if isfinite(path.cost):
        paths.paths[paths.count] = path
        paths.count += 1

    # LLL
    path.type = DUBINS_PATH_TYPE.LLL
    path.r_p = r_pl
    path.v_p = vp
    path.r_t = r_tl

    sub_v(r_tl, r_pl, v)
    div_v(v, 2, v)

    cross(v, n, Lv)
    normalize(Lv, Lv)
    mul_v(Lv, sqrt(4*args.r*args.r - dot(v, v)), Lv)

    sub_v(Lv, v, path.v_t)
    normalize(path.v_t, path.v_t)
    add_v(Lv, v, path.v_c)
    normalize(path.v_c, path.v_c)

    mul_v(path.v_c, 2*args.r, path.r_c)
    add_v(path.r_c, r_pl, path.r_c)

    negate_v(n, n)
    negate_v(vp, vp)
    path.theta_p = -angle(path.v_c, vp, n)
    negate_v(vp, vp)
    negate_v(n, n)
    negate_v(vt, vt)
    path.theta_t = -angle(path.v_t, vt, n)
    negate_v(vt, vt)
    negate_v(n, n)
    path.theta_c = angle(path.v_c, path.v_t, n)
    negate_v(n, n)

    path.cost = args.r * (abs(path.theta_p)+abs(path.theta_t)+abs(path.theta_c))
    negate_v(path.v_c, path.v_c)
    negate_v(path.v_p, path.v_p)
    
    if isfinite(path.cost):
        paths.paths[paths.count] = path
        paths.count += 1
    
    return paths

cpdef bint generate_path(double[:] t, double[:, :] out, dubins_path path):
    if path.type & DUBINS_PATH_TYPE.CSC:
        return _gen_csc(t, out, path)
    elif path.type & DUBINS_PATH_TYPE.CCC:
        return _gen_ccc(t, out, path)
    else:
        return False
        
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef bint _gen_csc(double[:] t, double[:, :] out, dubins_path p):
    cdef:
        double tt[2]
        double v_pn[3]
        double v_tn[3]
        double x
        int i, k
    
    if t.shape[0] > out.shape[0] or out.shape[1] < 3:
        return False
    
    cross(p.v_p, p.n, v_pn)
    cross(p.v_t, p.n, v_tn)
    
    lenl = sqrt(
        (p.l1[0]-p.l2[0])*(p.l1[0]-p.l2[0])
        +(p.l1[1]-p.l2[1])*(p.l1[1]-p.l2[1])
        +(p.l1[2]-p.l2[2])*(p.l1[2]-p.l2[2])
    )
    tt[0] = p.r*abs(p.theta_p)
    tt[1] = tt[0] + lenl
    for i in range(t.shape[0]):
        if t[i] < tt[0]:
            x = t[i]/p.r
            if p.theta_p < 0:
                x = -x
            for k in range(3):
                out[i][k] = p.r_p[k] + p.r*(p.v_p[k]*cos(x) + v_pn[k]*sin(x))
        elif t[i] > tt[1]:
            x = (t[i] - tt[1])/p.r
            if p.theta_t < 0:
                x = -x
            for k in range(3):
                out[i][k] = p.r_t[k] + p.r*(p.v_t[k]*cos(x) + v_tn[k]*sin(x))
        else:
            x = (t[i] - tt[0])/lenl
            for k in range(3):
                out[i][k] = p.l1[k]+x*(p.l2[k]-p.l1[k])
    return True

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef bint _gen_ccc(double[:] t, double[:, :] out, dubins_path p):
    cdef:
        double tt[2]
        double v_pn[3]
        double v_tn[3]
        double v_cn[3]
        double x
        int i, k
    
    if t.shape[0] > out.shape[0] or out.shape[1] < 3:
        return False
    
    cross(p.v_p, p.n, v_pn)
    cross(p.v_t, p.n, v_tn)
    cross(p.v_c, p.n, v_cn)
    
    tt[0] = p.r*abs(p.theta_p)
    tt[1] = tt[0] + p.r*abs(p.theta_c)
    for i in range(t.shape[0]):
        if t[i] < tt[0]:
            x = t[i]/p.r
            if p.theta_p < 0:
                x = -x
            for k in range(3):
                out[i][k] = p.r_p[k] + p.r*(p.v_p[k]*cos(x) + v_pn[k]*sin(x))
        elif t[i] > tt[1]:
            x = (t[i] - tt[1])/p.r
            if p.theta_t < 0:
                x = -x
            for k in range(3):
                out[i][k] = p.r_t[k] + p.r*(p.v_t[k]*cos(x) + v_tn[k]*sin(x))
        else:
            x = (t[i] - tt[0])/p.r
            if p.theta_c < 0:
                x = -x
            for k in range(3):
                out[i][k] = p.r_c[k] + p.r*(p.v_c[k]*cos(x) + v_cn[k]*sin(x))
    return True

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef bint generate_modified_path(
        double[:] t,
        double[:, :] out,
        dubins_path path,
        dubins_args args,
        double pitch_rate
    ):
    
    if t.shape[0] > out.shape[0] or out.shape[1] < 3:
        return False
    
    normalize(args.direction, args.direction)
    normalize(args.target_direction, args.target_direction)
    
    m0 = args.direction[2]
    m0 = m0 / sqrt(1 - m0*m0)
    m1 = args.target_direction[2]
    m1 = m1 / sqrt(1 - m1*m1)
    
    cdef quad q = solve_quad(path.cost, pitch_rate, m0, m1, args.position[2], args.target_position[2])
    
    if not generate_path(t, out, path):
        return False
    
    if not generate_quad(t, out[:,2], q):
        return False
    
    return True

cdef struct quad:
    double3 t[2]
    double3 a[2][3]
    double3 b[2]

@cython.boundscheck(False)
cpdef bint generate_quad(double[:] t, double[:] out, quad q):
    cdef int i
    if t.shape[0] > out.shape[0] or q.t[0] == q.t[1] == 0:
        return False
    for i in range(t.shape[0]):
        if t[i] < q.t[0]:
            out[i] = q.a[0][0]+t[i]*(q.a[0][1]+t[i]*q.a[0][2])
        elif t[i] > q.t[1]:
            out[i] = q.a[1][0]+t[i]*(q.a[1][1]+t[i]*q.a[1][2])
        else:
            out[i] = q.b[0]+q.b[1]*t[i]
    return True

# Auto-generated expressions for solving coeffs of the "dubins path"
# with parabolic altitude by Sympy and sympy.cse 
@cython.cdivision(True)
cpdef quad solve_quad(double delta_x, double acceleration, double m_0, double m_1, double x_0, double x_1):
    cdef:
        double x[60]
        quad q
    x[0] = m_1*m_1
    x[1] = (x_1 - x_0)*acceleration
    x[2] = 2*x[1]
    x[3] = x[0] + x[2]
    x[4] = delta_x*acceleration
    x[5] = 2*m_0
    x[6] = x[4]*x[5]
    x[7] = m_0*m_0
    x[8] = m_1*x[5]
    x[9] = x[7] - x[8]
    x[10] = -x[6] + x[9]
    x[11] = x[10] + x[3]
    x[12] = 1/acceleration
    x[13] = -m_0
    x[14] = m_1 + x[4]
    x[15] = x[13] + x[14]
    x[16] = 1/(2*x[15])
    x[17] = x[12]*x[16]
    x[18] = delta_x*delta_x
    x[19] = acceleration*acceleration*x[18]
    x[20] = 2*x[19]
    x[21] = m_1*x[4]
    x[22] = 4*x[21]
    x[23] = -x[2]
    x[24] = x[0] + x[23]
    x[25] = x[12]/8
    x[26] = -x[7]
    x[27] = acceleration/2
    x[28] = -x[27]
    x[29] = delta_x*m_1
    x[30] = x_1 - x[18]*x[27] - x[29]
    x[31] = -x[0]
    x[32] = -4*x[1]
    x[33] = 2*x[21]
    x[34] = x[19] + x[33]
    x[35] = sqrt(x[26] + x[31] + x[32] + x[34] + x[6] + x[8])
    x[36] = -x[35]
    x[37] = x[12]/2
    x[38] = x[37]*(x[15] + x[36])
    x[39] = x[37]*(x[15] + x[35])
    x[40] = -x[19]
    x[41] = x[2] - x[33]
    x[42] = -m_0*x[35] + m_1*x[35] + x[35]*x[4]
    x[43] = x[12]/4
    x[44] = m_0 + x[14]
    x[45] = x[6] + x[9]
    x[46] = sqrt(-x[0] - x[32] - x[33] - x[40] - x[45])
    x[47] = -m_1 + x[4]
    x[48] = m_0 + x[47]
    x[49] = x[37]*(-x[46] + x[48])
    x[50] = x[37]*(x[46] + x[48])
    x[51] = m_1*x[46]
    x[52] = m_0*x[46]
    x[53] = x[4]*x[46]
    x[54] = x[19] + x[41]
    x[55] = x_1 + acceleration*x[18]/2 - x[29]
    x[56] = -x[47]
    x[57] = x[24] + x[45]
    x[58] = 1/x[48]
    x[59] = x[37]*x[58]
    q.t[0] = -x[11]*x[17]
    q.t[1] = x[17]*(x[10] + x[20] + x[22] + x[24])
    q.b[0] = x_0+x[11]*x[11]*x[25]/(x[15]*x[15])
    q.b[1] = x[16]*(x[26] + x[3])
    q.a[0][0] = x_0
    q.a[0][1] = m_0
    q.a[0][2] = x[28]
    q.a[1][0] = x[30]
    q.a[1][1] = x[14]
    q.a[1][2] = x[28]
    if 0<q.t[0]<q.t[1]<delta_x:
        return q
    q.t[0] = x[38]
    q.t[1] = x[39]
    q.b[0] = x_0+x[43]*(x[40] + x[41] + x[42])
    q.b[1] = x[36]/2 + x[44]/2
    q.a[0][0] = x_0
    q.a[0][1] = m_0
    q.a[0][2] = x[27]
    q.a[1][0] = x[30]
    q.a[1][1] = x[14]
    q.a[1][2] = x[28]
    if 0<q.t[0]<q.t[1]<delta_x:
        return q
    q.t[0] = x[39]
    q.t[1] = x[38]
    q.b[0] = x_0-x[43]*(x[23] + x[34] + x[42])
    q.b[1] = x[35]/2 + x[44]/2
    q.a[0][0] = x_0
    q.a[0][1] = m_0
    q.a[0][2] = x[27]
    q.a[1][0] = x[30]
    q.a[1][1] = x[14]
    q.a[1][2] = x[28]
    if 0<q.t[0]<q.t[1]<delta_x:
        return q
    q.t[0] = x[49]
    q.t[1] = x[50]
    q.b[0] = x_0+x[43]*(x[51] - x[52] - x[53] + x[54])
    q.b[1] = m_0/2 + m_1/2 - x[4]/2 + x[46]/2
    q.a[0][0] = x_0
    q.a[0][1] = m_0
    q.a[0][2] = x[28]
    q.a[1][0] = x[55]
    q.a[1][1] = x[56]
    q.a[1][2] = x[27]
    if 0<q.t[0]<q.t[1]<delta_x:
        return q
    q.t[0] = x[50]
    q.t[1] = x[49]
    q.b[0] = x_0+x[43]*(-x[51] + x[52] + x[53] + x[54])
    q.b[1] = -x[13]/2 - x[46]/2 - x[47]/2
    q.a[0][0] = x_0
    q.a[0][1] = m_0
    q.a[0][2] = x[28]
    q.a[1][0] = x[55]
    q.a[1][1] = x[56]
    q.a[1][2] = x[27]
    if 0<q.t[0]<q.t[1]<delta_x:
        return q
    q.t[0] = -x[57]*x[59]
    q.t[1] = x[59]*(x[20] - x[22] + x[3] + x[45])
    q.b[0] = x_0-x[25]*(x[57]*x[57])/(x[48]*x[48])
    q.b[1] = x[58]*(x[2] + x[31] + x[7])/2
    q.a[0][0] = x_0
    q.a[0][1] = m_0
    q.a[0][2] = x[27]
    q.a[1][0] = x[55]
    q.a[1][1] = x[56]
    q.a[1][2] = x[27]
    if 0<q.t[0]<q.t[1]<delta_x:
        return q
    q.t[0] = 0
    q.t[1] = 0
    return q