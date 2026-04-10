/**
 * nmpfam_case_study.js — Overview page for the NMPFams case study.
 *
 * Fetches /api/nmpfam-case-study and renders:
 *   1. Summary stat cards (sampling, hits, intersections)
 *   2. Feature set intersection bar chart
 *   3. Activation distribution histogram
 *   4. Triple intersection feature cards (clickable)
 *   5. Broader GP+NMPFams table
 */

function fmtVal(v, d = 3) {
    if (v == null || v === undefined) return "\u2014";
    return typeof v === "number" ? v.toFixed(d) : String(v);
}

function createStatCard(title, inner) {
    const card = document.createElement("article");
    card.className = "stat-card";
    card.innerHTML = `<header><strong>${title}</strong></header><div>${inner}</div>`;
    return card;
}

// ── Stats cards ──

function renderStatsCards(container, summary) {
    container.innerHTML = "";

    const catBreakdown = Object.entries(summary.n_families_by_category || {})
        .map(([cat, n]) => `${cat}: ${n}`)
        .join(", ");

    container.appendChild(createStatCard("NMPFams Sample",
        `<div class="value">${summary.n_families_sampled} families</div>
         <div class="detail">${catBreakdown}</div>
         <div class="detail">All with AlphaFold2 structures</div>`
    ));

    container.appendChild(createStatCard("SAE Features with Hits",
        `<div class="value">${summary.n_features_with_nmpfam_hits} / ${summary.n_features_total}</div>
         <div class="detail">${(100 * summary.n_features_with_nmpfam_hits / summary.n_features_total).toFixed(1)}% of all features</div>
         <div class="detail">${summary.total_hit_instances} total hit instances</div>`
    ));

    container.appendChild(createStatCard("Geometry-Primary",
        `<div class="value">${summary.n_geometry_primary_with_nmpfam} / ${summary.n_geometry_primary}</div>
         <div class="detail">${(100 * summary.n_geometry_primary_with_nmpfam / Math.max(1, summary.n_geometry_primary)).toFixed(1)}% of confound-filtered features</div>
         <div class="detail">fire on novel metagenomic proteins</div>`
    ));

    container.appendChild(createStatCard("Triple Intersection",
        `<div class="value">${summary.n_triple_intersection}</div>
         <div class="detail">Sparse + Geometry-Primary + NMPFams</div>`
    ));

    const dist = summary.activation_distribution || {};
    container.appendChild(createStatCard("Activation Strength",
        `<div class="detail">&gt;75% of global max: <strong>${dist.gt_0_75 || 0}</strong> hits</div>
         <div class="detail">&gt;90% of global max: <strong>${dist.gt_0_90 || 0}</strong> hits</div>`
    ));
}

// ── Intersection bar chart ──

function renderVennChart(container, summary) {
    const sets = [
        { label: "All features", value: summary.n_features_total, color: "#cbd5e1" },
        { label: "NMPFams hits", value: summary.n_features_with_nmpfam_hits, color: "#60a5fa" },
        { label: "Sparse (<10%)", value: summary.n_sparse, color: "#a78bfa" },
        { label: "Geometry-primary", value: summary.n_geometry_primary, color: "#34d399" },
        { label: "Sparse + NMPFams", value: summary.n_sparse_with_nmpfam, color: "#818cf8" },
        { label: "GP + NMPFams", value: summary.n_geometry_primary_with_nmpfam, color: "#2dd4bf" },
        { label: "Triple intersection", value: summary.n_triple_intersection, color: "#f97316" },
    ];

    const traces = [{
        x: sets.map(s => s.label),
        y: sets.map(s => s.value),
        type: "bar",
        marker: { color: sets.map(s => s.color) },
        text: sets.map(s => String(s.value)),
        textposition: "outside",
        hovertemplate: "%{x}: %{y}<extra></extra>",
    }];

    Plotly.newPlot(container, traces, {
        title: { text: "Feature Set Sizes", font: { size: 14 } },
        yaxis: { title: "Number of features", type: "log" },
        height: 400,
        margin: { t: 50, b: 120, l: 60, r: 20 },
        xaxis: { tickangle: -30 },
    }, { responsive: true, displayModeBar: false });
}

// ── Activation distribution ──

function renderActivationChart(container, data) {
    // Collect all normalized activations from triple features
    const allNorms = [];
    for (const feat of [...(data.triple_features || []), ...(data.broader_gp_features || [])]) {
        allNorms.push(feat.top_nmpfam_norm_act);
    }

    // Also add from all features with hits (estimate from summary)
    const traces = [{
        x: allNorms,
        type: "histogram",
        xbins: { start: 0.5, end: 1.5, size: 0.05 },
        marker: { color: "#f97316" },
        name: "GP features",
    }];

    const shapes = [{
        type: "line", x0: 1.0, x1: 1.0, y0: 0, y1: 1,
        yref: "paper", line: { color: "#dc2626", width: 2, dash: "dash" },
    }];

    const annotations = [{
        x: 1.0, y: 1, yref: "paper", text: "SwissProt max",
        showarrow: true, arrowhead: 2, ax: 40, ay: -20,
        font: { color: "#dc2626", size: 11 },
    }];

    Plotly.newPlot(container, traces, {
        title: { text: "Peak Normalized Activation (GP Features with NMPFams Hits)", font: { size: 14 } },
        xaxis: { title: "Normalized activation (fraction of SwissProt global max)" },
        yaxis: { title: "Count" },
        shapes, annotations,
        height: 350,
        margin: { t: 50, b: 50, l: 60, r: 20 },
    }, { responsive: true, displayModeBar: false });
}

// ── Triple intersection feature cards ──

function renderTripleFeatures(container, features) {
    container.innerHTML = "";

    if (features.length === 0) {
        container.innerHTML = '<p class="secondary">No features in the triple intersection.</p>';
        return;
    }

    for (const feat of features) {
        const card = document.createElement("article");
        card.style.cursor = "pointer";
        card.style.transition = "box-shadow 0.2s";
        card.addEventListener("mouseenter", () => card.style.boxShadow = "0 2px 12px rgba(0,0,0,0.15)");
        card.addEventListener("mouseleave", () => card.style.boxShadow = "");
        card.addEventListener("click", () => window.location.href = `/nmpfam-case-study/${feat.feature_id}`);

        const geom = feat.geometry || {};
        const imp = geom.feature_importances || {};
        const topGeomFeat = Object.entries(imp).sort((a, b) => b[1] - a[1])[0];
        const topGeomName = topGeomFeat ? topGeomFeat[0].replace(/_/g, " ") : "\u2014";

        const ipro = feat.interpro || {};
        const iproName = ipro.protein_best_name || "None";

        // Radar glyph
        let radarHtml = "";
        if (imp && typeof aggregateToCategories === "function") {
            const scores = aggregateToCategories(imp);
            if (scores) {
                const radarDiv = document.createElement("div");
                radarDiv.style.cssText = "float:right;margin-left:1rem;";
                renderRadarGlyph(radarDiv, scores, { size: 80 });
                radarHtml = radarDiv.outerHTML;
            }
        }

        const exceeds = feat.top_nmpfam_norm_act > 1.0
            ? ' <span style="color:#dc2626;font-weight:bold">(exceeds SwissProt max!)</span>' : "";

        card.innerHTML = `
            ${radarHtml}
            <header>
                <strong>Feature ${feat.feature_id}</strong>
                <span class="secondary" style="float:right;font-size:0.85rem;">
                    ${feat.n_nmpfam_hits} NMPFams hit${feat.n_nmpfam_hits !== 1 ? "s" : ""}
                </span>
            </header>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;font-size:0.9rem;">
                <div>Coverage: <strong>${fmtVal(feat.coverage_pct, 1)}%</strong></div>
                <div>Composite: <strong>${fmtVal(feat.composite_score, 3)}</strong></div>
                <div>Top NMPFam act: <strong>${fmtVal(feat.top_nmpfam_norm_act, 3)}</strong>${exceeds}</div>
                <div>GBM AUC: <strong>${fmtVal(geom.gbm_auc_cv, 3)}</strong></div>
                <div>Top geometry: <strong>${topGeomName}</strong></div>
                <div>InterPro: <strong>${iproName}</strong></div>
            </div>
            <div style="margin-top:0.5rem;font-size:0.8rem;opacity:0.7;">
                ${feat.nmpfam_hits.slice(0, 3).map(h =>
                    `<a href="${h.nmpfams_url}" target="_blank" onclick="event.stopPropagation()">${h.family_id}</a> (${h.sequence_count} members, ${fmtVal(h.normalized_activation, 2)}x)`
                ).join(" &middot; ")}
            </div>
        `;
        container.appendChild(card);
    }
}

// ── Broader GP+NMPFams table ──

function renderBroaderTable(container, features) {
    if (features.length === 0) {
        container.innerHTML = '<p class="secondary">No additional geometry-primary features with NMPFams hits.</p>';
        return;
    }

    let html = '<table role="grid"><thead><tr>';
    html += '<th>Feature</th><th>Coverage</th><th>Composite</th><th>NMPFam Hits</th><th>Top Norm Act</th>';
    html += '</tr></thead><tbody>';
    for (const f of features) {
        html += `<tr style="cursor:pointer" onclick="window.location.href='/feature/${f.feature_id}'">`;
        html += `<td><a href="/feature/${f.feature_id}">${f.feature_id}</a></td>`;
        html += `<td>${fmtVal(f.coverage_pct, 1)}%</td>`;
        html += `<td>${fmtVal(f.composite_score, 3)}</td>`;
        html += `<td>${f.n_nmpfam_hits}</td>`;
        html += `<td>${fmtVal(f.top_nmpfam_norm_act, 3)}</td>`;
        html += '</tr>';
    }
    html += '</tbody></table>';
    container.innerHTML = html;
}

// ── Main ──

document.addEventListener("DOMContentLoaded", async () => {
    try {
        const res = await fetch("/api/nmpfam-case-study");
        if (!res.ok) throw new Error("NMPFams case study not built yet. Run build_nmpfam_case_study.py first.");
        const data = await res.json();
        const summary = data.summary || {};

        renderStatsCards(document.getElementById("stats-cards"), summary);
        renderVennChart(document.getElementById("venn-chart"), summary);
        renderActivationChart(document.getElementById("activation-chart"), data);
        renderTripleFeatures(document.getElementById("triple-container"), data.triple_features || []);
        renderBroaderTable(document.getElementById("broader-table"), data.broader_gp_features || []);
    } catch (err) {
        console.error(err);
        document.getElementById("stats-cards").innerHTML =
            `<p style="color:red">Error: ${err.message}</p>`;
    }
});
