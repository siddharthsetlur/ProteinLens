import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from functools import lru_cache
from pdb_plotter import ca_backbone, detect_alpha_helices_from_ca
from geometry.compute_geometric_features import *
from sklearn.decomposition import PCA

def list_batch_paths(batch_dir, first = 0, last = 21):
    batch_dir = Path(batch_dir)
    return [p for i in range(first, last + 1) if (p := batch_dir / f"batch_{i}.yaml").is_file()]

@lru_cache(maxsize=None)
def _load_yaml_cached(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def _normalize_pdb_text(pdb_text):
    return pdb_text.replace("\r\n", "\n").replace("\\\n", "")

def find_pdb_in_batches(entry_key, batch_paths):
    for path in batch_paths:
        data = _load_yaml_cached(path)
        if entry_key in data:
            val = data[entry_key]
            if isinstance(val, dict) and "pdb" in val and val["pdb"]:
                return _normalize_pdb_text(val["pdb"])
            print(f"Entry '{entry_key}' found in {path.name} but has no 'pdb' text.")
            raise ValueError(f"Entry '{entry_key}' found in {path.name} but has no 'pdb' text.")
    raise KeyError(f"Entry '{entry_key}' not found in any batch YAML.")

def load_groups(groups_yaml_path: str | Path) -> dict[int, list[str]]:
    data = yaml.safe_load(Path(groups_yaml_path).read_text(encoding="utf-8")) or {}
    groups: dict[int, list[str]] = {}
    for k, v in data.items():
        if isinstance(k, str) and k.isdigit():
            k = int(k)
        groups[int(k)] = list(v or [])
    return groups

def build_dataset(
    groups_yaml_path: str | Path,
    batch_dir: str | Path,
    output_file: str | Path,
    chain_id: str | None = "A",
    first_batch: int = 0,
    last_batch: int = 21,
    skip_missing: bool = True
):
    output_file = Path(output_file)

    # Check if the output file already exists
    if output_file.exists():
        print(f"Dataset file '{output_file}' already exists. Loading data from file.")
        return np.load(output_file)  # Load the saved dataset

    groups = load_groups(groups_yaml_path)
    batch_paths = list_batch_paths(batch_dir, first_batch, last_batch)
    rows = []
    protein_cache = {}  # Cache to store computed values for each protein

    for gid in range(0, len(groups)):
        print(gid)
        accessions = groups[gid]
        if not accessions:
            continue
        for acc in accessions:
            try:
                if acc in protein_cache:
                    # Reuse cached values
                    print(f"Reusing cached values for: {acc}")
                    rows.append([gid] + protein_cache[acc])
                    continue

                pdb_text = find_pdb_in_batches(acc, tuple(batch_paths))
                print(f"Found: {acc}")
                ca = ca_backbone(pdb_text, chain_id=chain_id)
                helices = detect_alpha_helices_from_ca(ca)
                if ca is None:
                    continue

                # Compute features
                wr_d = writhe(ca, ca)
                wr = np.sum(wr_d)
                _v2 = float(vassiliev(wr_d))
                cur = average_curvature(ca)
                tor = average_torsion(ca)
                ga = gyration_asphericity(ca)
                kink = kink_index(ca)
                p_m, p_s, d_m, d_s = helix_statistics(ca, helices)
                rog = radius_of_gyration(ca)
                planar = local_planarity_score(ca)
                end = end_to_end_distance(ca)
                ta, bc = helical_consistency(ca)
                L = float(len(ca))

                # Store computed values in cache.
                protein_cache[acc] = [acc, wr, _v2, cur, tor, kink, ga, p_m, p_s, d_m, d_s, rog, planar, end, ta, bc, L]

                # Append to rows
                rows.append([gid] + protein_cache[acc])
            except Exception as e:
                if not skip_missing:
                    raise
                # else skip
                continue

    data = np.array(rows)

    # Save the computed data to the output file
    np.save(output_file, data)
    print(f"Dataset saved to '{output_file}'.")

    return data

def cohens_d(x, y):
    # x = group values, y = rest
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    nx, ny = len(x), len(y)
    # pooled std
    sp = np.sqrt(((nx-1)*vx + (ny-1)*vy) / max(nx+ny-2, 1))
    return (mx - my) / (sp + 1e-12)

def binned_kl(p, q, eps=1e-9):
    # KL(p||q) for normalized hist vectors
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    return float(np.sum(p * np.log((p + eps) / (q + eps))))

def rank_groups_by_distinctiveness(values, groups, nbins=30):
    """
    values: 1D array (e.g., writhe_per_res OR curvature OR torsion)
    groups: 1D int array of same length
    Returns a list of (group_id, n, d, kl) sorted by |d| * KL.
    """
    valid = np.isfinite(values)
    vals = values[valid]; grps = groups[valid]
    # global hist for KL reference
    print(vals)
    lo, hi = np.nanpercentile(vals, [1, 99])  # robust range
    bins = np.linspace(lo, hi, nbins + 1)
    results = []
    for gid in np.unique(grps):
        gi = (grps == gid)
        xi = vals[gi]
        yi = vals[~gi]
        if len(xi) < 5 or len(yi) < 10:
            continue
        d = cohens_d(xi, yi)
        Hx, _ = np.histogram(xi, bins=bins)
        Hy, _ = np.histogram(yi, bins=bins)
        kl = binned_kl(Hx.astype(float), Hy.astype(float))
        score = abs(d) * kl  # combine separation & shape difference
        results.append((gid, len(xi), d, kl, score, bins))
    results.sort(key=lambda t: t[4], reverse=True)
    return results


def plot_topk_violins(values, groups, title, topk=12):
    ranks = rank_groups_by_distinctiveness(values, groups)
    top = ranks[:topk]
    if not top:
        print("No groups to plot.")
        return
    data = []
    labels = []
    for gid, n, d, kl, score, _ in top:
        data.append(values[groups == gid])
        labels.append(f"{gid}\n(n={n})")
    plt.figure(figsize=(max(8, 0.6*len(top)), 4))
    parts = plt.violinplot(data, showmeans=True, showextrema=False)
    plt.xticks(np.arange(1, len(labels)+1), labels, rotation=45, ha="right")
    plt.ylabel(title)
    plt.title(f"{title}: top-{len(top)} most distinctive nodes.")
    plt.tight_layout()
    plt.show()


def ecdf(x):
    x = np.sort(x[np.isfinite(x)])
    y = np.linspace(0, 1, len(x), endpoint=True)
    return x, y

def plot_topk_ecdfs(values, groups, title, topk=6):
    ranks = rank_groups_by_distinctiveness(values, groups)
    top = ranks[:topk]
    plt.figure(figsize=(7,5))

    x_all, y_all = ecdf(values[np.isfinite(values)])
    plt.plot(x_all, y_all, lw=2, alpha=0.3, label="All")
    for gid, n, d, kl, score, _ in top:
        xi = values[groups == gid]
        xg, yg = ecdf(xi)
        plt.plot(xg, yg, lw=2, label=f"group {gid} (n={n}, d={d:.2f})")
    plt.xlabel(title); plt.ylabel("CDF")
    plt.title(f"{title}: CDF for top-{len(top)} nodes")
    plt.legend(frameon=False)
    plt.tight_layout(); plt.show()

data = build_dataset(
    groups_yaml_path="Per_feature_max_examples.yaml",
    batch_dir="results",
    output_file="protein_dataset.npy",
    chain_id=None,
    first_batch=0,
    last_batch=21,
    skip_missing=True
)

groups = data[:, 0]
writhes = data[:, 2].astype(float)
plt.hist(writhes)
plt.show()

v2s = data[:, 3].astype(float)
curv = data[:, 4].astype(float)
tors = data[:, 5].astype(float)
kinks = data[:, 6].astype(float)
ga = data[:, 7].astype(float)
p_m = data[:, 8].astype(float)
p_s = data[:, 9].astype(float)
d_m = data[:, 10].astype(float)
d_s = data[:, 11].astype(float)
rog = data[:, 12].astype(float)
planar = data[:, 13].astype(float)
end = data[:, 14].astype(float)
t_a = data[:, 15].astype(float)
b_c = data[:, 16].astype(float)

# plot_topk_ecdfs(writhes, groups, "Writhe", topk=6)
# plot_topk_ecdfs(v2s, groups, "V2", topk=6)
# plot_topk_ecdfs(curv, groups, "Average curvature", topk=6)
# plot_topk_ecdfs(tors, groups, "Average torsion", topk=6)
# plot_topk_ecdfs(kinks, groups, "Kink index", topk=6)
# plot_topk_ecdfs(ga, groups, "Gyration asphericity", topk=6)
# plot_topk_ecdfs(p_m, groups, "Helix parallelism (mean)", topk=6)
# plot_topk_ecdfs(d_m, groups, "Helix distance (mean)", topk=6)
# plot_topk_ecdfs(rog, groups, "Radius of gyration", topk=6)
# plot_topk_ecdfs(planar, groups, "Planarity", topk=6)
# plot_topk_ecdfs(end, groups, "End-to-end distance", topk=6)
# plot_topk_ecdfs(t_a, groups, "Tangential alignment", topk=6)
# plot_topk_ecdfs(b_c, groups, "Binormal consistency", topk=6)

# plot_topk_ecdfs(ki, groups, "Kink index", topk=6)

def plot_pca_with_groups(data, groups, target):
    """
    Perform PCA on the data and plot the first two principal components, coloring points by their group.
    """
     
    pc1 = 0
    pc2 = 1
    # RM nans
    valid_rows = ~np.isnan(data).any(axis=1)
    data = data[valid_rows]
    groups = groups[valid_rows]
    # Perform PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data)

    # Generate a colormap for the groups
    unique_groups = np.unique(groups)
    for i in unique_groups:
        if 1400<int(i)<1600:
            print(i)
    print(unique_groups)
    colors = plt.cm.Set1([0 if x != f'{target}' else 1 for x in unique_groups])

    # Plot non targets first
    plt.figure(figsize=(8, 6))
    for i, group in enumerate(unique_groups):
        if group != f'{target}':
            group_indices = (groups == group)
            plt.scatter(
                pca_result[group_indices, pc1],
                pca_result[group_indices, pc2],
                alpha=0.75,
                s=30,
                color=colors[i],
                edgecolor='k'
            )
    
    # Plot targets after
    for i, group in enumerate(unique_groups):
        if group == f'{target}':
            group_indices = (groups == group)
            plt.scatter(
                pca_result[group_indices, pc1],
                pca_result[group_indices, pc2],
                alpha=1,
                s=60,
                color=colors[i],
                edgecolor='k',
                label = f'target:{target}'
            )

    # 2. Compute average pairwise distance within each group
    avg_within_dist = {}

    for group in unique_groups:
        idx = (groups == group)
        X = pca_result[idx, :2]  # use second two PCs
        n = X.shape[0]
        if n < 5:
            # Hard penalty or skip
            avg_within_dist[group] = np.inf
            continue

        if n < 2:
            # no within-group distance definable; treat as very large or skip
            avg_within_dist[group] = np.inf
            continue

        # pairwise distances (upper triangle only)
        diffs = X[:, None, :] - X[None, :, :]        # shape (n, n, 2)
        dists = np.sqrt((diffs ** 2).sum(axis=-1))   # shape (n, n)
        iu = np.triu_indices(n, k=1)
        mean_dist = dists[iu].mean()
        centroid = X.mean(axis=0)
        r = np.linalg.norm(centroid)

        avg_within_dist[group] = mean_dist/(r)

    # 3. Find most compact group
    sorted_groups = sorted(avg_within_dist.items(), key=lambda x: x[1])

    # 4. Print top 10 most compact groups
    print("Top 10 most compact groups (smallest avg within-group distance):")
    for group, dist in sorted_groups[:100]:
        if np.isinf(dist):
            continue  # skip groups where distance couldn't be computed
        print(f"{group}: {dist:.4f}")

    plt.xlabel(f"Principal Component {pc1}")
    plt.ylabel(f"Principal Component {pc2}")
    plt.title("PCA Total Geometry")
    plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

groups = data[:, 0]
print(groups)
# features = data[:, 8:11].astype(float)
features = data[:, 8:13].astype(float)
plot_pca_with_groups(features, groups, 6770) 
# cluster the decoder vector and label with colour scales on thresholded writhe/other invs, hopefully should cluster
# sae = load_sae_from_hf(plm_model="esm2-8m", plm_layer=4)

# figures: (above).
# Figure from their console.
# Where on sequence is feature being activated.

# check sequence similarity BLAST
# 