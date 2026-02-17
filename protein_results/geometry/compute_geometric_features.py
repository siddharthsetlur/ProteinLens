### This file will compute geometric features for each input carbon backbone

import numpy as np
from numba import prange, njit

'''
Writhe computations a.la. Klenin et. al: Computation of writhe in modeling of supercoiled DNA
With additional Fulton-MacPherson compactification for the configurational space integrals
The above is essential to let us compute higher order interactions with much higher accuracy. (Vassiliev invs)
'''

# Helpers: geometry primitives
# Keeping lots of comments to make everything concrete

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
        parallel_std = np.std(parallel_stats)

        return parallel_mean, parallel_std, dist_mean, dist_std
    
    else:
        return 0, 0, 0, 0

@njit()
def _safe_norm(v, eps=1e-12):
    '''
    Return norm but make it safe for any divisions
    '''
    n = np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    return n if n > eps else eps

@njit()
def _unit(v):
    '''
    Compute unit vector (v/|v|)
    '''
    n = _safe_norm(v)
    return v / n

@njit()
def ca_dihedral(p0, p1, p2, p3):
    '''
    Signed dihedral angle (radians) for four points
    '''
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1u = _unit(b1)

    v = b0 - np.dot(b0, b1u) * b1u # project b0 onto b1u
    w = b2 - np.dot(b2, b1u) * b1u # project b2 onto b1u

    x = np.dot(v, w) # angle between projected vectors
    y = np.dot(np.cross(b1u, v), w) 
    return np.arctan2(y, x) # dihedral angle

@njit()
def tangent_vectors(coords):
    '''
    Simply tangents
    '''
    n = coords.shape[0]
    T = np.zeros_like(coords)
    for i in range(n):
        if i == 0:
            d = coords[1] - coords[0]
        elif i == n - 1:
            d = coords[n-1] - coords[n-2]
        else:
            d = coords[i+1] - coords[i-1]
        T[i] = _unit(d)
    return T 

@njit()
def ca_curvature_profile(coords):
    '''
    Discrete curvature proxy per residue using 3 consecutive points:
    kappa_i = ||(t_{i-1} x t_i)|| / ds
    i.e. turning of tangents.
    '''
    n = coords.shape[0]
    T = tangent_vectors(coords)
    kappa = np.zeros(n, dtype=np.float64)
    for i in range(1, n-1):
        cp = np.cross(T[i-1], T[i])
        kappa[i] = np.sqrt(cp[0]*cp[0] + cp[1]*cp[1] + cp[2]*cp[2])
    return kappa

@njit()
def ca_torsion_profile(coords):
    '''
    Total torsion using dihedral formula
    '''
    n = coords.shape[0]
    tau = np.zeros(n, dtype=np.float64)
    for i in range(1, n-2):
        tau[i+1] = ca_dihedral(coords[i-1], coords[i], coords[i+1], coords[i+2])
    return tau

@njit()
def local_planarity_profile(coords, w=7):
    '''
    Should double check this.
    '''
    n = coords.shape[0]
    half = w // 2
    out = np.zeros(n, dtype=np.float64)
    for i in range(half, n-half):
        X = coords[i-half:i+half+1]
        c = np.zeros(3, dtype=np.float64)
        for k in range(X.shape[0]):
            c += X[k]
        c /= X.shape[0]
        # covariance
        C = np.zeros((3,3), dtype=np.float64)
        for k in range(X.shape[0]):
            v = X[k] - c
            C[0,0]+=v[0]*v[0]; C[0,1]+=v[0]*v[1]; C[0,2]+=v[0]*v[2]
            C[1,0]+=v[1]*v[0]; C[1,1]+=v[1]*v[1]; C[1,2]+=v[1]*v[2]
            C[2,0]+=v[2]*v[0]; C[2,1]+=v[2]*v[1]; C[2,2]+=v[2]*v[2]
        # eigvals (3x3): use numpy eigvals is fine in object mode; for njit keep simple:
        # We'll approximate by calling np.linalg.eigvals (numba supports for small arrays in many setups),
        # if it fails in your environment, move this function out of njit.
        vals = np.linalg.eigvals(C).real
        s = vals[0]+vals[1]+vals[2]
        m = vals[0]
        if vals[1] < m: m = vals[1]
        if vals[2] < m: m = vals[2]
        out[i] = (m/s) if s > 1e-15 else 0.0
    return out

# Sliding windows + pooling

def sliding_windows(n, w, stride=1):
    '''
    Windows (start, end) indices with end exclusive.
    '''
    for i in range(0, n - w + 1, stride):
        yield i, i + w

def window_pool(x, start, end, mode="mean", k=3):
    '''
    Pool values x[start:end] where x is 1D.
    mode: "mean", "max", "topk_mean" -> Here we take the output 1D signal of a protein and compute some local stats.
    '''
    seg = x[start:end]
    if seg.size == 0:
        return 0.0
    if mode == "mean":
        return float(np.mean(seg))
    if mode == "max":
        return float(np.max(seg))
    if mode == "topk_mean":
        kk = min(k, seg.size)
        # partial sort for speed
        idx = np.argpartition(seg, -kk)[-kk:]
        return float(np.mean(seg[idx]))

# Helix features: improve stats and bundle features

def helix_parallel_top_k(parallel_stats, k=3):
    '''
    Given list of dot products, return mean of top-k (most parallel by |dot|).
    '''
    if len(parallel_stats) == 0:
        return 0.0
    arr = np.abs(np.array(parallel_stats))
    kk = min(k, arr.size)
    idx = np.argpartition(arr, -kk)[-kk:]
    return float(np.mean(arr[idx]))

def fraction_parallel_vs_threshold(parallel_stats, thr=0.8):
    '''
    Fraction of dot above threshold; this way we can localise helix features
    '''
    if len(parallel_stats) == 0:
        return 0.0
    arr = np.abs(np.array(parallel_stats))
    return float(np.mean(arr >= thr))

def helix_crossing_angle_stats(parallel_stats):
    '''
    dot product to degrees for interpretability
    '''
    if len(parallel_stats) == 0:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.abs(np.array(parallel_stats))
    arr = np.clip(arr, 0.0, 1.0)
    theta = np.degrees(np.arccos(arr))
    return float(theta.mean()), float(theta.std()), float(np.mean(theta < 15.0)), float(np.mean(theta > 60.0))

def min_interhelix_distance(coords, helix_a, helix_b):
    '''
    Minimum Ca-Ca distance between two helix segments (start,end).
    '''

    a0,a1 = helix_a
    b0,b1 = helix_b
    A = coords[a0:a1]
    B = coords[b0:b1]
    # brute force, helices are small so this is ok
    # This is just finding the minimum helix dist
    mind = 1e9
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            d = np.linalg.norm(A[i] - B[j])
            if d < mind:
                mind = d
    return float(mind)

def helix_statistics_contact_filtered(coords, helices, contact_ca_dist=10.0):
    '''
    Like helix_statistics but only uses helix pairs that are in contact.
    Returns a bunch of stats:
      parallel_mean, parallel_std, dist_mean, dist_std,
      parallel_top3, frac_parallel_0p8,
      angle_mean, angle_std, angle_frac_lt15, angle_frac_gt60,
      n_helices, n_contact_pairs
    '''
    if len(helices) == 0:
        return (0,0,0,0,0,0,0,0,0,0,0,0)

    helix_stats = []
    for (s,e) in helices: # check helix struct
        helix = coords[s:e]
        centroid = helix.mean(axis=0) # com
        X = helix - centroid # vec
        C = (X.T @ X) / max(1, helix.shape[0]) # Covariance matrix
        vals, vecs = np.linalg.eigh(C) # eigen vals, eigen vecs
        axis = vecs[:, np.argmax(vals)] # choose eigenvect with max eigen val -> this is just the helix axis
        if np.dot(helix[-1] - helix[0], axis) < 0:
            axis = -axis
        helix_stats.append((centroid, axis, (s,e)))

    parallel_stats = []
    dist_stats = []
    n_contact = 0

    # run over all helix axis extracted as above

    for i in range(len(helix_stats)-1):
        for j in range(i+1, len(helix_stats)):
            (c1, a1, seg1) = helix_stats[i]
            (c2, a2, seg2) = helix_stats[j]
            # contact filter using min Ca distance (more specific than centroid)
            mind = min_interhelix_distance(coords, seg1, seg2)
            if mind <= contact_ca_dist:
                n_contact += 1
                parallel_stats.append(float(np.dot(a1, a2)))
                dist_stats.append(float(np.linalg.norm(c1 - c2)))

    if len(parallel_stats) == 0:
        # helices exist but no contacting pairs
        return (0,0,0,0,0,0,0,0,0,0, len(helices), 0)

    parallel_mean = float(np.mean(parallel_stats))
    parallel_std  = float(np.std(parallel_stats))
    dist_mean = float(np.mean(dist_stats))
    dist_std  = float(np.std(dist_stats))

    parallel_top3 = helix_parallel_top_k(parallel_stats, k=3)
    frac_par_0p8  = fraction_parallel_vs_threshold(parallel_stats, thr=0.8)
    ang_mean, ang_std, ang_lt15, ang_gt60 = helix_crossing_angle_stats(parallel_stats)

    return (parallel_mean, parallel_std, dist_mean, dist_std,
            parallel_top3, frac_par_0p8,
            ang_mean, ang_std, ang_lt15, ang_gt60,
            len(helices), n_contact)

# Turn / hairpin / strand-ish (Ca-only)

def turn_density(coords, w=7, curvature_thr=0.6):
    '''
    Fraction of centers whose local curvature exceeds threshold.
    Threshold should be tuned using your dataset distributions.
    '''
    kappa = ca_curvature_profile(coords)
    # ignore ends a bit
    if kappa.size < w:
        return 0.0
    half = w//2
    centers = kappa[half:-half]
    return float(np.mean(centers > curvature_thr))

def hairpin_score(coords, w=17):
    '''
    Simple beta-hairpin proxy:
      - window end-to-end distance small relative to contour length
      - tangent reversal across window
    Returns average score over windows (0..1ish).
    '''
    n = coords.shape[0]
    if n < w:
        return 0.0
    T = tangent_vectors(coords)
    scores = []
    for s,e in sliding_windows(n, w, stride=1):
        seg = coords[s:e]
        # contour length
        contour = 0.0
        for i in range(seg.shape[0]-1):
            contour += np.linalg.norm(seg[i+1]-seg[i])

        ee = np.linalg.norm(seg[-1]-seg[0]) # end to end distance

        compact = 1.0 - min(1.0, ee / max(1e-8, contour))  # close ends -> higher
        # this is scoring distance of end points versus contour, if contour is long and loops back - hairpin
        

        # tangent reversal (start vs end)
        rev = 0.5 * (1.0 - np.dot(T[s], T[e-1]))  # 0 if aligned, 1 if opposite
        # direction of hairpin.

        scores.append(compact * rev)
    return float(np.mean(scores)) if len(scores) else 0.0

def extended_fraction(coords, align_thr=0.9, curvature_thr=0.2):
    '''
    Strand proxy: straight (high tangent alignment) AND low curvature.
    '''
    T = tangent_vectors(coords)
    kappa = ca_curvature_profile(coords)
    n = coords.shape[0]
    if n < 3:
        return 0.0
    ok = 0
    tot = 0
    for i in range(n-1):
        dot = np.dot(T[i], T[i+1])
        if dot > align_thr and kappa[i] < curvature_thr:
            ok += 1
        tot += 1
    return float(ok / max(1, tot))


# Chirality / handedness features

def signed_torsion(coords):
    '''
    mean torsion (dihedral), std torsion,
    frac positive, frac negative
    '''

    tau = ca_torsion_profile(coords)
    # ignore ends where tau is 0 by construction
    core = tau[2:-2] if tau.size > 4 else tau
    if core.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = float(np.mean(core))
    std  = float(np.std(core))
    frac_pos = float(np.mean(core > 0))
    frac_neg = float(np.mean(core < 0))
    return mean, std, frac_pos, frac_neg

def dihedral_sign_consistency(coords, w=9):
    '''
    Fraction of windows where the majority dihedral sign is consistent.
    High when torsion sign doesn't flip often (helices tend to be consistent).
    '''

    tau = ca_torsion_profile(coords)
    n = tau.size
    if n < w:
        return 0.0
    half = w//2
    good = 0
    tot = 0
    for i in range(half, n-half):
        seg = tau[i-half:i+half+1]
        # ignore near-zero
        pos = np.sum(seg > 1e-6)
        neg = np.sum(seg < -1e-6)
        if pos + neg == 0:
            continue
        maj = max(pos, neg) / (pos + neg)
        if maj >= 0.8:
            good += 1
        tot += 1
    return float(good / max(1, tot))


# Local/windowed summaries (for correlation per window)

def local_curvature(coords, w=21, stride=1, pool="mean"):
    kappa = ca_curvature_profile(coords)
    vals = []
    for s,e in sliding_windows(len(kappa), w, stride):
        vals.append(window_pool(kappa, s, e, mode=pool))
    return np.array(vals, dtype=np.float64)

def local_torsion(coords, w=21, stride=1, pool="mean"):
    tau = ca_torsion_profile(coords)
    vals = []
    for s,e in sliding_windows(len(tau), w, stride):
        vals.append(window_pool(tau, s, e, mode=pool))
    return np.array(vals, dtype=np.float64)

def local_planarity(coords, w=21, stride=1, inner_w=7, pool="mean"):
    pl = local_planarity_profile(coords, w=inner_w)
    vals = []
    for s,e in sliding_windows(len(pl), w, stride):
        vals.append(window_pool(pl, s, e, mode=pool))
    return np.array(vals, dtype=np.float64)

def local_writhe(coords, w=41, stride=3):
    """
    Expensive. Computes writhe of each subchain with itself.
    Use larger w and stride. Returns array of writhe sums per window.
    """
    n = coords.shape[0]
    if n < w:
        return np.zeros(0, dtype=np.float64)
    out = []
    for s,e in sliding_windows(n, w, stride):
        seg = coords[s:e]
        wr_mat = writhe(seg, seg)
        # sum upper triangle to avoid double count
        out.append(float(np.sum(wr_mat)))
    return np.array(out, dtype=np.float64)


# Secondary structure things

def helix_segments(coords, helices):
    '''
    Return simple helix summary features:
      n_helices, helix_fraction, mean_helix_len, std_helix_len, max_helix_len
    '''
    n = coords.shape[0]
    if len(helices) == 0 or n == 0:
        return 0, 0.0, 0.0, 0.0, 0
    lens = np.array([e - s for (s,e) in helices], dtype=np.float64)
    helix_res = float(np.sum(lens))
    return (int(len(helices)),
            float(helix_res / n),
            float(lens.mean()),
            float(lens.std()),
            int(lens.max()))
