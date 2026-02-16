### This file will compute geometric features for each input carbon backbone

import numpy as np
from numba import prange, njit

'''
Writhe computations a.la. Klenin et. al: Computation of writhe in modeling of supercoiled DNA
With additional Fulton-MacPherson compactification for the configurational space integrals
The above is essential to let us compute higher order interactions with much higher accuracy. (Vassiliev invs)
'''

@njit()
def writhe(ring1, ring2):
    '''
    General computation over all segments.
    '''

    matrix = np.zeros((ring1.shape[0],ring2.shape[0]))
    for i in prange(ring1.shape[0]):
        for j in prange(ring2.shape[0]):
            matrix[i,j] = compute_kernel_chord(ring1, ring2, i, j)
    return matrix

@njit()
def FM_compactification(one, two, three, four, rho_factor=1e-3):
    '''
    Fulton MacPherson compactification. Should motivate the theory a bit I think.
    '''

    length_one = np.linalg.norm(two-one)
    length_two = np.linalg.norm(four-three)
    # Either a==d or b==c (can't have the edge points of a meet c for example - according to the formulation here).
    rho = rho_factor * max(1e-12, min(length_one, length_two))
    tau_1 = (two-one)/length_one # this is whats determining the signage below btw
    tau_2 = (four-three)/length_two

    if np.allclose(two, three):
        two_compact = two - rho * tau_1
        three_compact = three + rho * tau_2
        return one, two_compact, three_compact, four
    
    if np.allclose(one, four):
        one_compact = one + rho * tau_1
        four_compact = four - rho * tau_2
        return one_compact, two, three, four_compact
    
    else:
        return one, two, three, four
    
@njit()    
def vec_cross(vec_1, vec_2):
    '''
    Automated vector cross product checks.
    '''

    n = np.cross(vec_1,vec_2)
    if np.all(n) != 0.0:
        norm_n = np.linalg.norm(n)
        if norm_n > 1e-10:
            n /= norm_n
        else:
            n = np.array((0.0, 0.0, 0.0), dtype=np.float64)
    return n

@njit()
def clip1n1(x):
    return np.minimum(1.0, np.maximum(-1.0, x))

@njit()
def compute_kernel_chord(ring1, ring2, i, j):
    '''
    Klenin computation of writhe
    '''

    wr = 0 
    P = ring1.shape[0]
    one = ring1[np.mod(i-1,ring1.shape[0]),:]
    three = ring2[np.mod(j-1,ring2.shape[0]),:]
    two = ring1[np.mod(i,ring1.shape[0]),:]
    four = ring2[np.mod(j,ring2.shape[0]),:]

    if i == j: 
        return 0.0
    
    if (j-i)%P in (1, P-1):
        # Build Fulton-Macpherson compactification on edges which share vertices.
        # https://www.jstor.org/stable/pdf/2946631.pdf
        one, two, three, four = FM_compactification(one, two, three, four)

    # Standard Klenin techniques
    r12=two-one
    r34=four-three
    r23=three-two
    r13=three-one
    r14=four-one
    r24=four-two

    n1 = vec_cross(r13, r14)
    n2 = vec_cross(r14, r24)
    n3 = vec_cross(r24, r23)
    n4 = vec_cross(r23, r13)

    n1n2=clip1n1(np.dot(n1,n2))
    n2n3=clip1n1(np.dot(n2,n3))
    n3n4=clip1n1(np.dot(n3,n4))
    n4n1=clip1n1(np.dot(n4,n1))

    triple = float(np.dot(np.cross(r34,r12),r13))
    sign = 0.0 if abs(triple) < 1e-18 else np.sign(triple)

    omega = (np.arcsin(n1n2) + np.arcsin(n2n3) + np.arcsin(n3n4) + np.arcsin(n4n1)) * sign
    wr+=omega/(4*np.pi)

    return wr

@njit()
def vassiliev(wr):
    v2 = 0.0
    for i in range(0, len(wr)):
        for j in range(i, len(wr)):
            for k in range(j, len(wr)):
                for l in range(k, len(wr)):
                    v2 += wr[i][k]*wr[j][l]

    return v2

def average_torsion(coords):
    '''
    Torsion
    '''
    coords = np.array(coords)
    n = len(coords)
    if n < 4:
        raise ValueError("At least 4 points are required to compute torsion.")

    # Compute first, second, and third derivatives (finite differences)
    first_derivative = np.gradient(coords, axis=0)
    second_derivative = np.gradient(first_derivative, axis=0)
    third_derivative = np.gradient(second_derivative, axis=0)

    # Compute torsion at each point
    torsion = []
    for i in range(n):
        cross_product = np.cross(first_derivative[i], second_derivative[i])
        numerator = np.dot(cross_product, third_derivative[i])
        denominator = np.linalg.norm(cross_product) ** 2
        if denominator != 0:
            torsion.append(numerator / denominator)
        else:
            torsion.append(0)

    # Return the average torsion
    return np.mean(torsion)

def average_curvature(coords):
    '''
    Computes the average curvature of a 3D curve defined by coords.
    '''
    coords = np.array(coords)
    n = len(coords)

    # Compute first and second derivatives (finite differences)
    first_derivative = np.gradient(coords, axis=0)
    second_derivative = np.gradient(first_derivative, axis=0)

    # Compute curvature at each point
    curvature = []
    for i in range(n):
        cross_product = np.cross(first_derivative[i], second_derivative[i])
        numerator = np.linalg.norm(cross_product)
        denominator = np.linalg.norm(first_derivative[i]) ** 3
        if denominator != 0:
            curvature.append(numerator / denominator)
        else:
            curvature.append(0)

    # Return the average curvature
    return np.mean(curvature)

def end_to_end_distance(coords):
    return np.linalg.norm(coords[-1] - coords[0])

def radius_of_gyration(coords):
    c = coords.mean(axis=0)
    diffs = coords - c
    return np.sqrt((diffs * diffs).sum() / coords.shape[0])

def gyration_asphericity(coords):
    """
    Asphericity (b) from gyration tensor eigenvalues λ1≥λ2≥λ3:
    """
    c = coords.mean(axis=0)
    X = coords - c
    G = np.zeros((3,3))
    for i in range(coords.shape[0]):
        v = X[i]
        G[0,0] += v[0]*v[0]; G[0,1] += v[0]*v[1]; G[0,2] += v[0]*v[2]
        G[1,0] += v[1]*v[0]; G[1,1] += v[1]*v[1]; G[1,2] += v[1]*v[2]
        G[2,0] += v[2]*v[0]; G[2,1] += v[2]*v[1]; G[2,2] += v[2]*v[2]
    G /= coords.shape[0]
    vals = np.linalg.eigvals(G).real
    for i in range(3):
        for j in range(i+1,3):
            if vals[j] > vals[i]:
                tmp = vals[i]; vals[i] = vals[j]; vals[j] = tmp
    return vals[0] - 0.5*(vals[1] + vals[2])


def tangent_vectors(coords):
    d = np.gradient(coords, axis=0)
    n = coords.shape[0]
    T = np.zeros_like(coords)
    for i in range(n):
        norm = np.linalg.norm(d[i])
        T[i] = d[i]/norm if norm > 1e-15 else np.array((0.0,0.0,0.0))
    return T

def local_planarity_score(coords, w=7):
    """
    For each center i, PCA of window [i-w//2, i+w//2].
    Score = smallest eigenvalue / sum(eigs). Lower means more planar.
    """
    n = coords.shape[0]
    half = w//2
    acc = 0.0; cnt = 0
    for i in range(half, n-half):
        X = coords[i-half:i+half+1]
        c = X.mean(axis=0)
        Xc = X - c
        C = np.zeros((3,3))
        for k in range(Xc.shape[0]):
            v = Xc[k]
            C[0,0]+=v[0]*v[0]; C[0,1]+=v[0]*v[1]; C[0,2]+=v[0]*v[2]
            C[1,0]+=v[1]*v[0]; C[1,1]+=v[1]*v[1]; C[1,2]+=v[1]*v[2]
            C[2,0]+=v[2]*v[0]; C[2,1]+=v[2]*v[1]; C[2,2]+=v[2]*v[2]
        vals = np.linalg.eigvals(C).real
        s = vals[0]+vals[1]+vals[2]
        # find min
        m = vals[0]
        if vals[1] < m: m = vals[1]
        if vals[2] < m: m = vals[2]
        score = (m/s) if s>1e-15 else 0.0
        acc += score; cnt += 1
    return acc/max(1,cnt)

def helical_consistency(coords):
    """
    A simple 'helixiness' proxy from tangent vectors:
    average T_i · T_{i+1}  (close to 1: straight/helix; lower: bends/turns)
    plus average signed binormal rotation consistency.
    """
    T = tangent_vectors(coords)
    n = T.shape[0]
    # tangent alignment
    s1 = 0.0; c1 = 0
    for i in range(n-1):
        s1 += np.dot(T[i], T[i+1]); c1 += 1
    # binormal rotation
    s2 = 0.0; c2 = 0
    for i in range(1, n-1):
        b1 = np.cross(T[i-1], T[i]);  nb1 = np.linalg.norm(b1);  b1 = b1/nb1 if nb1>1e-15 else b1
        b2 = np.cross(T[i], T[i+1]);  nb2 = np.linalg.norm(b2);  b2 = b2/nb2 if nb2>1e-15 else b2
        s2 += np.dot(b1, b2); c2 += 1
    return s1/max(1,c1), s2/max(1,c2)  # (tangent alignment, binormal consistency)

def kink_index(coords, theta_deg=60.0):
    """
    Fraction of positions where the turning angle between consecutive segments exceeds theta.
    """
    T = tangent_vectors(coords)
    thr = np.cos(theta_deg*np.pi/180.0)
    n = T.shape[0]
    cnt = 0; tot = 0
    for i in range(n-1):
        dot = np.dot(T[i], T[i+1])
        if dot < thr:
            cnt += 1
        tot += 1
    return cnt/max(1,tot)

def pairwise_statistics(coords):
    return coords

def helix_statistics(coords, helices):

    helix_stats = []
    if len(helices)>0:

        for x in helices:
            helix = coords[x[0]:x[1]]
            centroid = helix.mean(axis=0) 

            X = helix - centroid

            C = (X.T @ X) / helix.shape[0]

            vals, vecs = np.linalg.eigh(C)
            axis = vecs[:, np.argmax(vals)]

            # make axis point N->C
            if np.dot(helix[-1] - helix[0], axis) < 0:
                axis = -axis

            helix_detail = [centroid, axis]
            helix_stats.append(helix_detail)

        parallel_stats = []
        dist_stats = []
        for H1 in range(0, len(helix_stats)-1):
            for H2 in range(H1+1, len(helix_stats)):
                parallel = np.dot(helix_stats[H1][1], helix_stats[H2][1])
                parallel_stats.append(parallel)
                dist_stats.append(np.linalg.norm(helix_stats[H1][0]-helix_stats[H2][0]))

        dist_mean = np.mean(dist_stats)
        dist_std = np.std(dist_stats)
        parallel_mean = np.mean(parallel_stats)
        parallel_std = np.mean(parallel_stats)

        return parallel_mean, parallel_std, dist_mean, dist_std
    
    else:
        return 0, 0, 0, 0
