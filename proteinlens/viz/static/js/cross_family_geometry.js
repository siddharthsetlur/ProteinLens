/**
 * cross_family_geometry.js — Overview page for the cross-family geometry case study.
 *
 * Fetches /api/cross-family-geometry and renders:
 *   1. Global stat cards
 *   2. Structural category bar chart
 *   3. Composite score vs InterPro F1 scatter
 *   4. Residue-level InterPro F1 histogram
 *   5. Cross-family feature cards (clickable)
 *   6. Full feature table
 */

function fmtVal(v, decimals = 3) {
    if (v === null || v === undefined) return "\u2014";
    return Number(v).toFixed(decimals);
}

function createStatCard(title, bodyHtml) {
    const card = document.createElement("article");
    card.className = "stat-card";
    card.innerHTML = `<header><strong>${title}</strong></header>${bodyHtml}`;
    return card;
}

// ============================================================
// 1. Global Stats
// ============================================================

function renderGlobalStats(container, stats, features) {
    container.innerHTML = "";

    const nt = stats.null_thresholds;
    const nSingle = stats.n_zero_single_family;

    container.appendChild(createStatCard("Geometry-Primary Features", `
        <div class="value">${stats.total_geometry_primary} / ${stats.n_features_with_geometry || "?"}</div>
        <div class="detail">${fmtVal(100 * stats.total_geometry_primary / (stats.n_features_with_geometry || 1), 1)}% of features with geometry data</div>
        <div class="detail">SAE latents where geometry is the primary encoding</div>
    `));

    container.appendChild(createStatCard("Cross-Family Features", `
        <div class="value">${stats.n_cross_family} (${stats.pct_cross_family}%)</div>
        <div class="detail">InterPro F1 0.3\u20130.7, multiple families</div>
        <div class="detail">Clearest evidence of family-transcendent geometry</div>
    `));

    container.appendChild(createStatCard("Multi-Family Coverage", `
        <div class="value">${stats.n_multi_family} / ${stats.total_geometry_primary}</div>
        <div class="detail">${stats.pct_multi_family}% have \u22652 families with F1 &gt; 0.5</div>
        <div class="detail"><strong>${nSingle} feature${nSingle === 1 ? "" : "s"}</strong> explained by a single family</div>
    `));

    container.appendChild(createStatCard("Residue-Level InterPro", `
        <div class="value">Max F1 = ${stats.interpro_residue_f1_max}</div>
        <div class="detail">Mean: ${stats.interpro_residue_f1_mean} \u00b7 Median: ${stats.interpro_residue_f1_median}</div>
        <div class="detail">All below null threshold (${fmtVal(nt.interpro_res_f1)})</div>
        <div class="detail">Family labels fail at the residue level</div>
    `));

    // Motif RMSD stats
    const rmsds = features
        .filter(f => f.motif_rmsd_per_pos !== null)
        .map(f => f.motif_rmsd_per_pos);
    const crossRmsds = features
        .filter(f => f.is_cross_family && f.motif_rmsd_per_pos !== null)
        .map(f => f.motif_rmsd_per_pos);

    container.appendChild(createStatCard("Mean Motif RMSD/pos", `
        <div class="value">${fmtVal(rmsds.reduce((a, b) => a + b, 0) / rmsds.length, 3)} \u00c5</div>
        <div class="detail">Cross-family: ${fmtVal(crossRmsds.reduce((a, b) => a + b, 0) / crossRmsds.length, 3)} \u00c5</div>
        <div class="detail">Tight structural consensus across fragments</div>
    `));

    container.appendChild(createStatCard("Null Thresholds (p95)", `
        <div class="detail">InterPro Res F1: ${fmtVal(nt.interpro_res_f1, 4)}</div>
        <div class="detail">Motif PR-AUC: ${fmtVal(nt.motif_pr_auc, 4)}</div>
        <div class="detail">Position F1: ${fmtVal(nt.position_f1, 4)}</div>
        <div class="detail">From ${nt.n_sparse_features || "?"} sparse features (&lt;1% activation)</div>
    `));
}

// ============================================================
// 1b. Methodology
// ============================================================

function renderMethodology(container, stats) {
    const m = stats.methodology;
    const nt = stats.null_thresholds;
    if (!m) {
        container.innerHTML = '<p class="secondary">Methodology data not available.</p>';
        return;
    }

    container.innerHTML = `
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;">
            <article class="info-card">
                <header><strong>Geometry-Primary Classification</strong></header>
                <p class="secondary" style="font-size:0.85rem;margin-bottom:0.5rem;">
                    A feature is classified as <strong>geometry-primary</strong> if all four conditions hold:
                </p>
                <ol style="font-size:0.85rem;margin:0;padding-left:1.2rem;">
                    ${m.classification_criteria.map(c => `<li style="margin-bottom:0.3rem;">${c}</li>`).join("")}
                </ol>
                <p class="secondary" style="font-size:0.85rem;margin-top:0.75rem;margin-bottom:0;">
                    <strong>Composite score</strong> = <code>${m.composite_score_formula}</code>
                </p>
            </article>
            <article class="info-card">
                <header><strong>Null Distribution</strong></header>
                <p class="secondary" style="font-size:0.85rem;margin-bottom:0.5rem;">
                    ${m.null_distribution_method}
                </p>
                <table role="grid" style="font-size:0.85rem;margin:0;">
                    <thead><tr><th>Metric</th><th>Null p95</th></tr></thead>
                    <tbody>
                        <tr><td>InterPro Residue F1</td><td><strong>${fmtVal(nt.interpro_res_f1, 4)}</strong></td></tr>
                        <tr><td>Sequence Motif PR-AUC</td><td><strong>${fmtVal(nt.motif_pr_auc, 4)}</strong></td></tr>
                        <tr><td>Position F1</td><td><strong>${fmtVal(nt.position_f1, 4)}</strong></td></tr>
                        <tr><td>Geometry PR-AUC</td><td><strong>${fmtVal(stats.geom_pr_auc_threshold, 4)}</strong> (fixed)</td></tr>
                    </tbody>
                </table>
            </article>
        </div>
        <div style="margin-top:1rem;">
            <article class="info-card">
                <header><strong>Cross-Family Identification</strong></header>
                <p class="secondary" style="font-size:0.85rem;margin-bottom:0;">
                    ${m.cross_family_criteria}. These features fire on the same geometric motif
                    across evolutionarily unrelated protein families &mdash; InterPro partially
                    matches because each family happens to contain the geometry, but no single
                    family explains the feature.
                </p>
            </article>
        </div>
    `;
}

// ============================================================
// 2. Structural Category Chart
// ============================================================

function renderStructuralCategoryChart(div, allCats, crossCats) {
    const categories = Object.keys(allCats);
    const allCounts = categories.map(c => allCats[c] || 0);
    const crossCounts = categories.map(c => crossCats[c] || 0);

    const traces = [
        {
            x: categories, y: allCounts, name: "All geometry-primary",
            type: "bar", marker: { color: "#2563eb" },
        },
        {
            x: categories, y: crossCounts, name: "Cross-family subset",
            type: "bar", marker: { color: "#dc2626" },
        },
    ];

    const layout = {
        barmode: "group",
        xaxis: { tickangle: -35, tickfont: { size: 10 } },
        yaxis: { title: "Number of features" },
        height: 400,
        margin: { t: 20, b: 150, l: 60, r: 20 },
        legend: { x: 0.6, y: 0.95 },
    };

    Plotly.newPlot(div, traces, layout, { responsive: true, displayModeBar: false });
}

// ============================================================
// 3. Scatter: Composite vs InterPro F1
// ============================================================

function renderScatter(div, features) {
    const cross = features.filter(f => f.is_cross_family);
    const other = features.filter(f => !f.is_cross_family);

    const makeTrace = (data, name, color, symbol) => ({
        x: data.map(f => f.best_interpro_protein_f1),
        y: data.map(f => f.composite_score),
        text: data.map(f => `Feature ${f.feature_id}<br>${f.structural_category}<br>` +
            `${f.best_interpro_protein_name}<br>${f.n_families_above_03} families >0.3`),
        name, type: "scatter", mode: "markers",
        marker: { color, size: 8, symbol, opacity: 0.8 },
        hovertemplate: "%{text}<extra></extra>",
    });

    const traces = [
        makeTrace(other, "Geometry-primary", "#2563eb", "circle"),
        makeTrace(cross, "Cross-family", "#dc2626", "diamond"),
        {
            x: [0.3, 0.3], y: [0, 1], name: "F1 = 0.3",
            type: "scatter", mode: "lines",
            line: { color: "#999", dash: "dash", width: 1 },
            showlegend: false,
        },
        {
            x: [0.7, 0.7], y: [0, 1], name: "F1 = 0.7",
            type: "scatter", mode: "lines",
            line: { color: "#999", dash: "dash", width: 1 },
            showlegend: false,
        },
    ];

    // Annotation for the cross-family zone
    const layout = {
        xaxis: { title: "Best InterPro Protein F1", range: [0, 1.05] },
        yaxis: { title: "Geometry Composite Score", range: [0, 1.05] },
        height: 450,
        margin: { t: 20, b: 50, l: 60, r: 20 },
        legend: { x: 0.01, y: 0.99 },
        shapes: [{
            type: "rect", xref: "x", yref: "paper",
            x0: 0.3, x1: 0.7, y0: 0, y1: 1,
            fillcolor: "rgba(220,38,38,0.05)",
            line: { width: 0 },
        }],
        annotations: [{
            x: 0.5, y: 1.02, xref: "x", yref: "y",
            text: "Cross-family zone",
            showarrow: false,
            font: { size: 11, color: "#dc2626" },
        }],
    };

    Plotly.newPlot(div, traces, layout, { responsive: true, displayModeBar: false });
}

// ============================================================
// 4. Residue-level F1 histogram
// ============================================================

function renderResidueF1Histogram(div, features, nullThreshold) {
    const vals = features.map(f => f.best_interpro_residue_f1);

    const traces = [{
        x: vals, type: "histogram",
        marker: { color: "#2563eb" },
        nbinsx: 30,
        name: "Residue F1",
    }];

    const layout = {
        xaxis: { title: "Best InterPro Residue F1" },
        yaxis: { title: "Count" },
        height: 350,
        margin: { t: 20, b: 50, l: 60, r: 20 },
        shapes: [{
            type: "line", xref: "x", yref: "paper",
            x0: nullThreshold, x1: nullThreshold, y0: 0, y1: 1,
            line: { color: "#dc2626", dash: "dash", width: 2 },
        }],
        annotations: [{
            x: nullThreshold, y: 1, xref: "x", yref: "paper",
            text: `Null p95 = ${fmtVal(nullThreshold)}`,
            showarrow: true, arrowhead: 0, ax: 50, ay: -20,
            font: { color: "#dc2626", size: 11 },
        }],
    };

    Plotly.newPlot(div, traces, layout, { responsive: true, displayModeBar: false });
}

// ============================================================
// 5. Cross-family feature cards
// ============================================================

function renderCrossFamilyCards(container, features) {
    container.innerHTML = "";
    const crossFeatures = features
        .filter(f => f.is_cross_family)
        .sort((a, b) => b.composite_score - a.composite_score);

    const grid = document.createElement("div");
    grid.style.display = "grid";
    grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(450px, 1fr))";
    grid.style.gap = "1rem";

    for (const f of crossFeatures) {
        const card = document.createElement("article");
        card.className = "info-card";
        card.style.cursor = "pointer";
        card.style.transition = "box-shadow 0.2s";
        card.addEventListener("mouseenter", () => { card.style.boxShadow = "0 2px 12px rgba(0,0,0,0.15)"; });
        card.addEventListener("mouseleave", () => { card.style.boxShadow = ""; });
        card.addEventListener("click", () => {
            window.location.href = `/cross-family-geometry/${f.feature_id}`;
        });

        const familyList = f.interpro_families
            .filter(fam => fam.f1 > 0.3)
            .slice(0, 5)
            .map(fam => `<tr>
                <td style="font-size:0.8rem">${fam.name}</td>
                <td>${fmtVal(fam.f1)}</td>
                <td>${fmtVal(fam.precision)}</td>
                <td>${fmtVal(fam.recall)}</td>
            </tr>`)
            .join("");

        card.innerHTML = `
            <header>
                <strong>Feature ${f.feature_id}</strong>
                <span class="secondary" style="margin-left:0.5rem;font-size:0.85rem">Composite: ${fmtVal(f.composite_score)}</span>
            </header>
            <div style="margin:0.4rem 0;">
                <span class="badge badge-done">${f.structural_category}</span>
                <span class="badge badge-count" style="font-family:monospace">${f.top_geometric_feature}</span>
                <span class="badge badge-count">${f.n_families_above_03} families</span>
            </div>
            <div class="detail" style="margin-bottom:0.4rem;">
                GBM AUC: ${fmtVal(f.gbm_auc_cv)} \u00b7
                Concordance F1: ${fmtVal(f.concordance_f1)} \u00b7
                Motif RMSD/pos: ${fmtVal(f.motif_rmsd_per_pos)} \u00c5
            </div>
            <div class="detail" style="margin-bottom:0.4rem;">
                InterPro protein F1: ${fmtVal(f.best_interpro_protein_f1)} \u00b7
                InterPro residue F1: ${fmtVal(f.best_interpro_residue_f1)} \u00b7
                Seq motif F1: ${fmtVal(f.motif_seq_f1)}
            </div>
            <table role="grid" style="font-size:0.8rem;margin:0;">
                <thead><tr><th>Family</th><th>F1</th><th>Prec</th><th>Rec</th></tr></thead>
                <tbody>${familyList}</tbody>
            </table>
        `;

        grid.appendChild(card);
    }

    container.appendChild(grid);
}

// ============================================================
// 6. All features table
// ============================================================

function renderAllFeaturesTable(container, features) {
    const rows = features
        .sort((a, b) => b.composite_score - a.composite_score)
        .map(f => `<tr style="${f.is_cross_family ? 'background:#fff5f5;' : ''}">
            <td><a href="/cross-family-geometry/${f.feature_id}">${f.feature_id}</a></td>
            <td>${f.is_cross_family ? '<span class="badge badge-done" style="font-size:0.7rem">cross-family</span>' : ''}</td>
            <td>${fmtVal(f.composite_score)}</td>
            <td style="font-size:0.8rem">${f.structural_category}</td>
            <td style="font-family:monospace;font-size:0.8rem">${f.top_geometric_feature}</td>
            <td>${fmtVal(f.gbm_auc_cv)}</td>
            <td>${fmtVal(f.concordance_f1)}</td>
            <td>${fmtVal(f.best_interpro_protein_f1)}</td>
            <td>${f.n_families_above_03}</td>
            <td>${fmtVal(f.best_interpro_residue_f1)}</td>
            <td>${fmtVal(f.motif_rmsd_per_pos)}</td>
            <td>${fmtVal(f.motif_seq_f1)}</td>
        </tr>`)
        .join("");

    container.innerHTML = `
        <table role="grid" style="font-size:0.8rem;">
            <thead><tr>
                <th>ID</th><th>Type</th><th>Composite</th><th>Struct. Cat.</th>
                <th>Top Geom</th><th>GBM AUC</th><th>Conc F1</th>
                <th>IP Prot F1</th><th>#Fam</th><th>IP Res F1</th>
                <th>RMSD/pos</th><th>Motif F1</th>
            </tr></thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}

// ============================================================
// Main
// ============================================================

document.addEventListener("DOMContentLoaded", async () => {
    try {
        const res = await fetch("/api/cross-family-geometry");
        if (!res.ok) throw new Error(`Failed: ${res.status}`);
        const data = await res.json();

        const stats = data.global_stats;
        const features = data.features;

        renderMethodology(document.getElementById("methodology-container"), stats);

        renderGlobalStats(document.getElementById("global-stats"), stats, features);

        renderStructuralCategoryChart(
            document.getElementById("struct-cat-chart"),
            data.structural_categories_all,
            data.structural_categories_cross_family
        );

        renderScatter(document.getElementById("scatter-chart"), features);

        renderResidueF1Histogram(
            document.getElementById("residue-f1-chart"),
            features,
            stats.null_thresholds.interpro_res_f1
        );

        renderCrossFamilyCards(document.getElementById("cross-family-container"), features);

        renderAllFeaturesTable(document.getElementById("all-features-table"), features);

    } catch (err) {
        console.error(err);
        document.getElementById("global-stats").innerHTML =
            `<p style="color:red">Error: ${err.message}. Run <code>python scripts/build_cross_family_case_study.py --data-dir &lt;dir&gt;</code> first.</p>`;
    }
});
