/**
 * subdomain_detail.js — deep-dive for a single subdomain family.
 *
 * URL: /subdomain-decomposition/{source}/{code}
 *   source = "interpro" or "cath"
 *   code   = InterPro accession (e.g. IPR003593) or CATH label (e.g. 1.10.760.10)
 *
 * Fetches /api/subdomain-case-study for the group structure, then
 * /api/feature/{id}/geometry for each member to pull the 44-d importance
 * vector. Renders a heatmap (rows=features, cols=geometry descriptors),
 * a cosine similarity heatmap, and a feature table.
 */

const GEOM_COS_THRESHOLD = 0.5;

// Map raw geometry feature names to their radar category. `RADAR_CATEGORIES`
// is provided by /static/js/radar_glyph.js (loaded before this file).
const FEATURE_TO_CATEGORY = {};
const CATEGORY_ORDER = [];
if (typeof RADAR_CATEGORIES !== "undefined") {
    for (const cat of RADAR_CATEGORIES) {
        CATEGORY_ORDER.push(cat.name);
        for (const feat of cat.features) FEATURE_TO_CATEGORY[feat] = cat.name;
    }
}

function fmt(v, d = 3) { return v == null ? "—" : Number(v).toFixed(d); }

function parseUrl() {
    const parts = window.location.pathname.split("/").filter(Boolean);
    // ["subdomain-decomposition", "<source>", "<...code parts...>"]
    if (parts.length < 3) return { source: null, code: null };
    const source = parts[1];
    const code = decodeURIComponent(parts.slice(2).join("/"));
    return { source, code };
}

function findGroup(sc, source, code) {
    const list = source === "cath" ? (sc.cath_groups || [])
              : source === "interpro" ? (sc.interpro_groups || sc.groups || [])
              : [];
    return list.find((g) => String(g.annotation_code) === String(code));
}

async function fetchGeometryVector(fid) {
    const res = await fetch(`/api/feature/${fid}/geometry`).catch(() => null);
    if (!res || !res.ok) return {};
    const geo = await res.json();
    const rl = (geo.geometric_residue_level) || {};
    return rl.feature_importances || {};
}

function renderSummary(group, source, feats) {
    document.getElementById("page-title").textContent = group.annotation_name || group.annotation_code;
    document.title = `${group.annotation_code} — subdomain family`;
    const info = document.getElementById("family-info");
    const shown = group.n_features_shown != null && group.n_features_shown < group.n_features
        ? ` (showing ${group.n_features_shown})`
        : "";
    info.innerHTML = `
        <strong>${source === "cath" ? "CATH" : "InterPro"}-residue family</strong>
        · <code>${group.annotation_code}</code>
        · ${group.n_features} features${shown}
        · mean residue F1 ${fmt(group.mean_residue_f1)}
        · mean geometry PR-AUC ${fmt(group.mean_geom_pr_auc)}
        · mean cosine similarity <strong>${fmt(group.mean_cosine_similarity)}</strong>
    `;

    // Distinguishability flags (same logic as list page)
    const hasSig = (f, k) => f[`q_${k}`] != null && f[`q_${k}`] < 0.05;
    const distinctLabels = (values) => {
        const s = new Set();
        for (const v of values) {
            if (v == null) continue;
            s.add(v);
            if (s.size > 1) return true;
        }
        return false;
    };
    const geomDist = group.mean_cosine_similarity != null
        ? group.mean_cosine_similarity < GEOM_COS_THRESHOLD
        : distinctLabels(feats.map((f) => f.top_geometric_feature));
    // MEME / position distinguishability: use q-values from the group's own records
    // (feature records store q_motif_pr_auc / q_position_f1 alongside the score).
    const memeDist = distinctLabels(feats.filter((f) => f.q_motif_pr_auc != null && f.q_motif_pr_auc < 0.05).map((f) => f.best_seq_motif));
    const posLabels = feats.filter((f) => f.q_position_f1 != null && f.q_position_f1 < 0.05).map((f) => f.position_f1_label);
    const posDist = distinctLabels(posLabels);

    const flag = (label, on, color) =>
        `<span class="q-chip ${on ? "sig" : ""}" style="${on ? `background:${color};color:#fff;` : ""}">${label}</span>`;
    document.getElementById("flags-strip").innerHTML = `
        ${flag(`Geometry-distinguishable${group.mean_cosine_similarity != null ? ` (cos < ${GEOM_COS_THRESHOLD})` : ""}`, geomDist, "#10b981")}
        ${flag("MEME-distinguishable", memeDist, "#6366f1")}
        ${flag("Position-distinguishable", posDist, "#eab308")}
    `;
}

function renderImportanceHeatmap(container, feats, importanceVectors) {
    // Union of feature names across members, ordered by radar category then alphabetically.
    const allNames = new Set();
    for (const v of importanceVectors) Object.keys(v).forEach((k) => allNames.add(k));
    const catOrderIndex = (name) => {
        const cat = FEATURE_TO_CATEGORY[name];
        const idx = cat ? CATEGORY_ORDER.indexOf(cat) : -1;
        return idx === -1 ? CATEGORY_ORDER.length : idx;
    };
    const names = Array.from(allNames).sort((a, b) => {
        const ia = catOrderIndex(a), ib = catOrderIndex(b);
        if (ia !== ib) return ia - ib;
        return a.localeCompare(b);
    });

    const z = feats.map((_, i) =>
        names.map((n) => importanceVectors[i][n] || 0)
    );
    const y = feats.map((f) => `f/${f.feature_id}`);
    const colWidths = names.length;

    Plotly.newPlot(container, [{
        z, x: names, y,
        type: "heatmap",
        colorscale: [[0, "#ffffff"], [0.5, "#c7d2fe"], [1, "#3730a3"]],
        hovertemplate: "%{y}<br>%{x}: %{z:.3f}<extra></extra>",
        colorbar: { title: "Importance", thickness: 10, len: 0.8 },
    }], {
        margin: { t: 10, r: 20, b: 140, l: 90 },
        xaxis: { tickangle: -60, tickfont: { size: 9 } },
        yaxis: { tickfont: { size: 10 }, autorange: "reversed" },
        paper_bgcolor: "#fff",
        plot_bgcolor: "#fff",
        height: Math.max(300, 24 * feats.length + 200),
    }, { responsive: true, displayModeBar: false });
}

function renderCosineHeatmap(container, feats, matrix) {
    if (!matrix || matrix.length === 0) {
        container.innerHTML = '<p class="secondary">No cosine matrix available.</p>';
        return;
    }
    const ids = feats.map((f) => `f/${f.feature_id}`);
    Plotly.newPlot(container, [{
        z: matrix,
        x: ids, y: ids,
        type: "heatmap",
        zmin: 0, zmax: 1,
        colorscale: [[0, "#ffffff"], [0.5, "#a5b4fc"], [1, "#312e81"]],
        hovertemplate: "%{y} vs %{x}: %{z:.3f}<extra></extra>",
        colorbar: { title: "cos", thickness: 10, len: 0.8 },
    }], {
        margin: { t: 10, r: 20, b: 80, l: 80 },
        xaxis: { tickfont: { size: 10 }, tickangle: -60 },
        yaxis: { tickfont: { size: 10 }, autorange: "reversed" },
        paper_bgcolor: "#fff",
        plot_bgcolor: "#fff",
        height: Math.max(260, 24 * feats.length + 120),
    }, { responsive: true, displayModeBar: false });
}

function renderFeatureList(container, feats, importanceVectors) {
    const rows = feats.map((f, i) => {
        const vec = importanceVectors[i] || {};
        const top = Object.entries(vec)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5)
            .map(([name, val]) => `<span style="font-family:monospace">${name}</span>: ${Number(val).toFixed(3)}`)
            .join(" · ");
        const rmsd = f.motif_rmsd_per_pos != null
            ? `${Number(f.motif_rmsd_per_pos).toFixed(2)} Å/pos`
            : (f.motif_rmsd != null ? `${Number(f.motif_rmsd).toFixed(2)} Å` : "—");
        return `<tr onclick="location.href='/feature/${f.feature_id}'" style="cursor:pointer">
            <td style="font-weight:600">${f.feature_id}</td>
            <td>${f.structural_category || "—"}</td>
            <td style="font-family:monospace;font-size:.78rem">${f.top_geometric_feature || "—"}</td>
            <td>${fmt(f.geom_pr_auc)}</td>
            <td style="font-size:.78rem">${rmsd}</td>
            <td style="font-size:.78rem">${f.pct_proteins_activated != null ? Number(f.pct_proteins_activated).toFixed(1) + "%" : "—"}</td>
            <td style="font-size:.78rem;color:#374151">${top || "—"}</td>
        </tr>`;
    }).join("");
    container.innerHTML = `
        <table style="width:100%;border-collapse:collapse;font-size:.82rem">
            <thead style="background:#f9fafb">
                <tr><th>Feature</th><th>Structural category</th><th>Top geom feature</th><th>Geom PR-AUC</th><th>Motif RMSD</th><th>% prot.</th><th>Top-5 importances</th></tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}

document.addEventListener("DOMContentLoaded", async () => {
    const { source, code } = parseUrl();
    if (!source || !code) {
        document.getElementById("page-title").textContent = "Invalid URL";
        document.getElementById("family-info").textContent = "Expected /subdomain-decomposition/{interpro|cath}/{code}.";
        return;
    }
    try {
        const scRes = await fetch("/api/subdomain-case-study");
        if (!scRes.ok) throw new Error(`/api/subdomain-case-study: ${scRes.status}`);
        const sc = await scRes.json();
        const group = findGroup(sc, source, code);
        if (!group) {
            document.getElementById("page-title").textContent = "Family not found";
            document.getElementById("family-info").textContent =
                `No ${source}-residue group with code ${code}. It may have dropped out under the sparsity filter or group-cap.`;
            return;
        }
        const feats = group.features || [];

        // Fetch 44-d importance vectors per feature in parallel.
        const vectors = await Promise.all(feats.map((f) => fetchGeometryVector(f.feature_id)));

        renderSummary(group, source, feats);
        renderImportanceHeatmap(document.getElementById("heatmap-container"), feats, vectors);
        renderCosineHeatmap(document.getElementById("cosine-container"), feats, group.cosine_matrix);
        renderFeatureList(document.getElementById("feature-list-container"), feats, vectors);
    } catch (err) {
        console.error("subdomain detail load failed:", err);
        document.getElementById("family-info").innerHTML = `<p style="color:red">Error: ${err.message}</p>`;
    }
});
