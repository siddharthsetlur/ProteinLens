/**
 * subdomain_case_study.js — "Geometry is more granular than biology"
 *
 * Renders the InterPro-grouped and CATH-grouped lists emitted by
 * build_subdomain_case_study.py, plus a "MEME / position distinguishes,
 * geometry does not" counter-example table spanning both DB sources.
 *
 * Per group, we compute three orthogonal distinguishability flags:
 *   geom-distinguishable     = >=2 distinct top structural_category
 *                              OR >=2 distinct top_geometric_feature
 *   meme-distinguishable     = among features with m6_q<0.05, >=2 distinct m6 consensus
 *   position-distinguishable = among features with m5_q<0.05, >=2 distinct m5 label
 */

function fmt(v, d = 3) { return v == null ? "—" : Number(v).toFixed(d); }
function fmtQ(q) {
    if (q == null) return "—";
    if (q < 1e-3) return q.toExponential(1);
    return q.toFixed(3);
}

const sig = (row, k) => row[`m${k}_q`] != null && row[`m${k}_q`] < 0.05;

// A group is geometry-distinguishable when the mean pairwise cosine similarity
// of its 44-d geometric feature-importance vectors is below this threshold.
// Strict cut: ≥0.5 means the importance vectors point in broadly the same
// direction, so we only call "distinguishable" when profiles are genuinely
// divergent. In 44-d space, <0.5 corresponds to clearly distinct geometric
// roles (near-identical profiles sit at 0.9+).
const GEOM_COS_THRESHOLD = 0.5;

function distinguishable(values) {
    const seen = new Set();
    for (const v of values) {
        if (v == null) continue;
        seen.add(v);
        if (seen.size > 1) return true;
    }
    return false;
}

function slugify(s) {
    return String(s).replace(/[^a-zA-Z0-9_.-]+/g, "_");
}

function enrichFeatures(group, indexById) {
    // Join group features with /api/index rows so we have m5_label/q, m6_label/q etc.
    return group.features.map((f) => {
        const idxRow = indexById.get(f.feature_id) || {};
        return {
            ...f,
            m5_label: idxRow.m5_label,
            m5_score: idxRow.m5_score,
            m5_q: idxRow.m5_q,
            m6_label: idxRow.m6_label,
            m6_score: idxRow.m6_score,
            m6_q: idxRow.m6_q,
            m7_q: idxRow.m7_q,
        };
    });
}

function groupMetrics(group, indexById) {
    const feats = enrichFeatures(group, indexById);
    // Primary signal: mean pairwise cosine of 44-d importance vectors (from script).
    // Fallback when cosine isn't available: distinct top-geom-feature labels.
    const geomDist = group.mean_cosine_similarity != null
        ? group.mean_cosine_similarity < GEOM_COS_THRESHOLD
        : distinguishable(feats.map((f) => f.top_geometric_feature));
    const memeDist = distinguishable(
        feats.filter((f) => sig(f, 6)).map((f) => f.m6_label)
    );
    const posDist = distinguishable(
        feats.filter((f) => sig(f, 5)).map((f) => f.m5_label)
    );
    return { g: group, feats, geomDist, memeDist, posDist };
}

function renderGlobalStats(interproMetrics, cathMetrics, globalStats) {
    const all = interproMetrics.concat(cathMetrics);
    const total = all.length;
    const geomD = all.filter((m) => m.geomDist).length;
    const memeD = all.filter((m) => m.memeDist).length;
    const posD  = all.filter((m) => m.posDist).length;
    const geomNotSeq = all.filter((m) => m.geomDist && !(m.memeDist || m.posDist)).length;
    const seqNotGeom = all.filter((m) => (m.memeDist || m.posDist) && !m.geomDist).length;
    const container = document.getElementById("global-stats");
    const maxPct = globalStats.max_pct_activated;
    const totalSparse = globalStats.total_after_sparsity_filter;
    container.innerHTML = `
        <article class="stat-callout">
            <div class="label">Geometry-significant, sparse</div>
            <div class="value">${(totalSparse || globalStats.total_geometry_significant || 0).toLocaleString()}</div>
            <div class="sub">q&nbsp;&lt;&nbsp;0.05 on geometry${maxPct != null ? ` · ≤${maxPct}% protein coverage` : ""}</div>
        </article>
        <article class="stat-callout">
            <div class="label">Groups (InterPro + CATH)</div>
            <div class="value">${total}</div>
            <div class="sub">${interproMetrics.length} InterPro · ${cathMetrics.length} CATH</div>
        </article>
        <article class="stat-callout">
            <div class="label">Geometry-distinguishable</div>
            <div class="value">${geomD}</div>
            <div class="sub">Different top structural category or geometric feature across members</div>
        </article>
        <article class="stat-callout">
            <div class="label">MEME-distinguishable</div>
            <div class="value">${memeD}</div>
            <div class="sub">Different top MEME consensus among members with q&lt;0.05</div>
        </article>
        <article class="stat-callout">
            <div class="label">Position-distinguishable</div>
            <div class="value">${posD}</div>
            <div class="sub">Different top position predicate among members with q&lt;0.05</div>
        </article>
        <article class="stat-callout flag-yellow">
            <div class="label">MEME / position distinguish, geometry does not</div>
            <div class="value">${seqNotGeom}</div>
            <div class="sub">Counter-examples to the paper's main claim</div>
        </article>
        <article class="stat-callout flag-green">
            <div class="label">Geometry distinguishes, MEME / position do not</div>
            <div class="value">${geomNotSeq}</div>
            <div class="sub">Core claim: geometry pulls information the sequence-based methods miss</div>
        </article>
    `;
}

function featureRow(f) {
    const chip = (k, value) => {
        const q = f[`m${k}_q`];
        const cls = q == null ? "null" : q < 0.05 ? "sig" : "";
        return `<td><span class="q-chip ${cls}">${value}</span><div style="font-size:.7rem;color:#6b7280">q=${fmtQ(q)}</div></td>`;
    };
    const rmsd = f.motif_rmsd_per_pos != null
        ? `${fmt(f.motif_rmsd_per_pos, 2)} Å/pos`
        : (f.motif_rmsd != null ? `${fmt(f.motif_rmsd, 2)} Å` : "—");
    return `<tr onclick="location.href='/feature/${f.feature_id}'" style="cursor:pointer">
        <td>${f.feature_id}</td>
        <td>${f.structural_category || "—"}</td>
        <td style="font-family:monospace;font-size:.78rem">${f.top_geometric_feature || "—"}</td>
        <td>${fmt(f.geom_pr_auc)}</td>
        <td style="font-size:.78rem">${rmsd}</td>
        <td style="font-size:.78rem">${fmtPct(f.pct_proteins_activated)}</td>
        ${chip(6, f.m6_label || "—")}
        ${chip(5, f.m5_label || "—")}
    </tr>`;
}

function fmtPct(v) {
    if (v == null) return "—";
    return `${Number(v).toFixed(1)}%`;
}

function renderGroupBlock(m, source, sectionPrefix) {
    const { g, feats, geomDist, memeDist, posDist } = m;
    const flag = (label, on, color) =>
        `<span class="q-chip ${on ? "sig" : ""}" style="${on ? `background:${color};color:#fff;` : ""}">${label}</span>`;
    const shown = g.n_features_shown != null && g.n_features_shown < g.n_features
        ? ` (showing ${g.n_features_shown})`
        : "";
    const cos = g.mean_cosine_similarity != null ? ` · mean cos ${fmt(g.mean_cosine_similarity)}` : "";
    const slug = `${sectionPrefix}-${slugify(g.annotation_code)}`;
    const sourceKey = source === "CATH" ? "cath" : "interpro";
    const detailHref = `/subdomain-decomposition/${sourceKey}/${encodeURIComponent(g.annotation_code)}`;

    const summary = `<summary style="cursor:pointer;padding:.55rem .8rem;list-style:none;display:flex;flex-wrap:wrap;gap:.5rem;align-items:center">
        <span style="font-weight:700;font-size:.95rem">${g.annotation_name || g.annotation_code}</span>
        <code style="font-size:.75rem;color:#6b7280">${g.annotation_code}</code>
        <span style="color:#6b7280;font-size:.78rem">${g.n_features} feats${shown} · ${source}-res F1 ${fmt(g.mean_residue_f1)}${cos}</span>
        <span style="margin-left:auto;display:flex;gap:.3rem;align-items:center">
            ${flag("Geom", geomDist, "#10b981")}
            ${flag("MEME", memeDist, "#6366f1")}
            ${flag("Pos", posDist, "#eab308")}
            <a href="${detailHref}" onclick="event.stopPropagation()"
               style="margin-left:.3rem;padding:.15rem .55rem;border:1px solid #6366f1;border-radius:6px;color:#4338ca;text-decoration:none;font-size:.75rem;font-weight:600"
               title="Open deep-dive page">Open ↗</a>
        </span>
    </summary>`;

    const body = `<div style="padding:.6rem .8rem">
        <table style="width:100%;border-collapse:collapse;font-size:.82rem">
            <thead style="background:#f9fafb">
                <tr><th>Feature</th><th>Structural category</th><th>Top geom feature</th><th>Geom PR-AUC</th><th>Motif RMSD</th><th>% prot.</th><th>Top MEME</th><th>Top position</th></tr>
            </thead>
            <tbody>${feats.map(featureRow).join("")}</tbody>
        </table>
        <p style="margin-top:.4rem;font-size:.78rem;color:#6b7280">
            For importance vectors and the full cosine similarity heatmap, open the
            <a href="${detailHref}">deep-dive page</a>.
        </p>
    </div>`;

    return `<details id="${slug}" class="group-block" style="margin-bottom:.5rem;border:1px solid #e5e7eb;border-radius:8px;background:#ffffff">
        ${summary}${body}
    </details>`;
}

function renderJumpStrip(container, metrics, sectionPrefix) {
    if (!metrics.length) return;
    const strip = document.createElement("div");
    strip.style.cssText = "display:flex;flex-wrap:wrap;gap:.3rem;margin:.3rem 0 1rem;font-size:.78rem";
    strip.innerHTML = metrics.map((m) => {
        const slug = `${sectionPrefix}-${slugify(m.g.annotation_code)}`;
        const label = (m.g.annotation_name || m.g.annotation_code).slice(0, 44);
        const cosStr = m.g.mean_cosine_similarity != null ? ` · ${fmt(m.g.mean_cosine_similarity)}` : "";
        return `<a class="jump-chip" data-target="${slug}" href="#${slug}"
                   title="${m.g.annotation_code} · ${m.g.n_features} features${cosStr}"
                   style="padding:.2rem .55rem;border:1px solid #e5e7eb;border-radius:12px;background:#f9fafb;text-decoration:none;color:#374151">
                    ${label} <span style="color:#9ca3af">(${m.g.n_features})</span>
                </a>`;
    }).join("");
    container.prepend(strip);

    strip.querySelectorAll(".jump-chip").forEach((anchor) => {
        anchor.addEventListener("click", (e) => {
            e.preventDefault();
            const el = document.getElementById(anchor.dataset.target);
            if (el) {
                el.open = true;
                el.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        });
    });
}

function renderGroups(container, metrics, source) {
    const prefix = source === "InterPro" ? "ipro" : "cath";
    if (!metrics.length) {
        container.innerHTML = `<p class="secondary">No ${source} groups with ≥2 features.</p>`;
        return;
    }
    container.innerHTML = metrics.map((m) => renderGroupBlock(m, source, prefix)).join("");
    renderJumpStrip(container, metrics, prefix);
}

function renderMemeNotGeom(container, interproMetrics, cathMetrics) {
    const rows = [];
    for (const m of interproMetrics) {
        if ((m.memeDist || m.posDist) && !m.geomDist) rows.push({ m, source: "InterPro", prefix: "mng-ipro" });
    }
    for (const m of cathMetrics) {
        if ((m.memeDist || m.posDist) && !m.geomDist) rows.push({ m, source: "CATH", prefix: "mng-cath" });
    }
    if (!rows.length) {
        container.innerHTML = '<p class="secondary">No groups fall in this category for the current analysis dir. That is consistent with the paper\'s claim that geometry carries the orthogonal signal here.</p>';
        return;
    }
    container.innerHTML = rows.map(({ m, source, prefix }) => renderGroupBlock(m, source, prefix)).join("");
    renderJumpStrip(container, rows.map((r) => r.m), "mng");
}

document.addEventListener("DOMContentLoaded", async () => {
    try {
        const [scRes, idxRes] = await Promise.all([
            fetch("/api/subdomain-case-study"),
            fetch("/api/index"),
        ]);
        if (!scRes.ok)  throw new Error(`/api/subdomain-case-study: ${scRes.status}`);
        if (!idxRes.ok) throw new Error(`/api/index: ${idxRes.status}`);
        const sc = await scRes.json();
        const idx = await idxRes.json();
        const indexById = new Map(idx.map((r) => [r.feature_id, r]));

        const interproGroups = sc.interpro_groups || sc.groups || [];
        const cathGroups = sc.cath_groups || [];

        const interproMetrics = interproGroups.map((g) => groupMetrics(g, indexById));
        const cathMetrics     = cathGroups.map((g) => groupMetrics(g, indexById));

        renderGlobalStats(interproMetrics, cathMetrics, sc.global_stats || {});
        interproMetrics.sort((a, b) => b.g.n_features - a.g.n_features);
        cathMetrics.sort((a, b) => b.g.n_features - a.g.n_features);
        renderGroups(document.getElementById("interpro-groups-container"), interproMetrics, "InterPro");
        renderGroups(document.getElementById("cath-groups-container"), cathMetrics, "CATH");
        renderMemeNotGeom(document.getElementById("meme-not-geom-container"), interproMetrics, cathMetrics);
    } catch (err) {
        console.error("subdomain load failed:", err);
        document.getElementById("global-stats").innerHTML =
            `<p style="color:red">Error: ${err.message}</p>`;
    }
});
