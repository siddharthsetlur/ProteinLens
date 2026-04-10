/**
 * subdomain_case_study.js — Sub-domain geometric decomposition case study.
 *
 * Shows how multiple geometry-primary SAE features decompose a single
 * InterPro annotation into distinct geometric sub-structures.
 */

function fmtVal(v, decimals = 3) {
    return v === null || v === undefined ? "\u2014" : Number(v).toFixed(decimals);
}

function createStatCard(title, bodyHtml) {
    const card = document.createElement("article");
    card.className = "stat-card";
    card.innerHTML = `<header><strong>${title}</strong></header>${bodyHtml}`;
    return card;
}

// ── Global stats cards ──

function renderGlobalStats(container, stats) {
    container.innerHTML = "";

    container.appendChild(createStatCard("Geometry-Primary Features",
        `<div class="value">${stats.total_geometry_primary}</div>
         <div class="detail">Features whose activation is best explained by 3D structure</div>`
    ));

    container.appendChild(createStatCard("With High InterPro Protein F1",
        `<div class="value">${stats.n_with_high_interpro_protein_f1} / ${stats.total_geometry_primary}</div>
         <div class="detail">${stats.pct_with_high_interpro_protein_f1}% have protein F1 &ge; ${stats.min_protein_f1_threshold}</div>
         <div class="detail">Fire on known protein families, but domain boundaries don't explain residue-level activation</div>`
    ));

    container.appendChild(createStatCard("Multi-Feature Annotations",
        `<div class="value">${stats.n_annotations_with_multiple_features}</div>
         <div class="detail">${stats.n_features_in_groups} GP features across ${stats.n_annotations_with_multiple_features} annotations</div>
         <div class="detail">Each decomposed into distinct geometric sub-structures</div>`
    ));
}

// ── Scatter: protein F1 vs residue F1 ──

function renderScatter(div, groups) {
    // Flatten all features from all groups
    const allFeats = groups.flatMap((g) =>
        g.features.map((f) => ({ ...f, annotation_name: g.annotation_name, annotation_code: g.annotation_code }))
    );

    // Color by structural category
    const categories = [...new Set(allFeats.map((f) => f.structural_category))];
    const colors = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
        "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
        "#ffd8b1", "#000075", "#a9a9a9",
    ];
    const catColor = {};
    categories.forEach((c, i) => { catColor[c] = colors[i % colors.length]; });

    const traces = categories.map((cat) => {
        const feats = allFeats.filter((f) => f.structural_category === cat);
        return {
            x: feats.map((f) => f.interpro_protein_f1),
            y: feats.map((f) => f.interpro_residue_f1),
            text: feats.map((f) => `Feature ${f.feature_id}<br>${f.annotation_name}<br>Geom PR-AUC: ${fmtVal(f.geom_pr_auc)}<br>Top: ${f.top_geometric_feature}`),
            customdata: feats.map((f) => f.feature_id),
            mode: "markers",
            type: "scatter",
            name: cat.length > 30 ? cat.slice(0, 28) + "\u2026" : cat,
            marker: { size: 8, color: catColor[cat], opacity: 0.8, line: { width: 0.5, color: "#333" } },
            hovertemplate: "%{text}<extra></extra>",
        };
    });

    Plotly.newPlot(div, traces, {
        title: { text: "InterPro Protein F1 vs Residue F1 (geometry-primary features)", font: { size: 13 } },
        xaxis: { title: { text: "InterPro Protein F1", font: { size: 11 } }, range: [0.5, 1.05], gridcolor: "#e9ecef" },
        yaxis: { title: { text: "InterPro Residue F1", font: { size: 11 } }, range: [-0.01, 0.25], gridcolor: "#e9ecef" },
        margin: { t: 45, r: 20, b: 50, l: 55 },
        hovermode: "closest",
        paper_bgcolor: "#fff",
        plot_bgcolor: "#f8f9fa",
        showlegend: true,
        legend: { font: { size: 9 }, bgcolor: "rgba(255,255,255,0.85)" },
        annotations: [{
            x: 0.85, y: 0.22, xref: "x", yref: "y",
            text: "High protein F1, low residue F1:<br>domain predicts <em>which proteins</em>,<br>not <em>which residues</em>",
            showarrow: false, font: { size: 10, color: "#666" },
            bgcolor: "rgba(255,255,255,0.8)", borderpad: 4,
        }],
    }, { responsive: true, displayModeBar: false });

    document.getElementById(div.id || div).on("plotly_click", (data) => {
        const fid = data.points[0].customdata;
        if (fid != null) window.location.href = `/feature/${fid}`;
    });
}

// ── Annotation group cards ──

function renderGroups(container, groups) {
    container.innerHTML = "";

    for (const group of groups) {
        const card = document.createElement("article");
        card.style.marginBottom = "1.5rem";

        // Header with annotation info
        const header = document.createElement("header");
        header.innerHTML = `
            <strong>${group.annotation_code}</strong> &mdash; ${group.annotation_name}
            <span class="badge badge-count" style="margin-left:0.5rem">${group.n_features} features</span>
            <span class="badge badge-done" style="margin-left:0.3rem">${group.n_distinct_categories} geometric categories</span>
        `;
        card.appendChild(header);

        // Summary line
        const summary = document.createElement("p");
        summary.className = "secondary";
        summary.style.fontSize = "0.85rem";
        summary.innerHTML = `
            Mean protein F1: ${fmtVal(group.mean_interpro_protein_f1)} &middot;
            Max residue F1: ${fmtVal(group.max_interpro_residue_f1)} &middot;
            Mean geom PR-AUC: ${fmtVal(group.mean_geom_pr_auc)} &middot;
            Categories: ${group.distinct_categories.join(", ")}
        `;
        card.appendChild(summary);

        // Feature table
        const table = document.createElement("table");
        table.style.fontSize = "0.8rem";
        table.style.width = "100%";
        table.innerHTML = `
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Structural Category</th>
                    <th>Top Geom. Feature</th>
                    <th>Geom PR-AUC</th>
                    <th>Prot. F1</th>
                    <th>Res. F1</th>
                    <th>CATH F1</th>
                    <th>Motif F1</th>
                    <th>Score</th>
                </tr>
            </thead>
            <tbody>
                ${group.features.map((f) => `
                    <tr style="cursor:pointer" onclick="window.location.href='/feature/${f.feature_id}'">
                        <td><strong>${f.feature_id}</strong></td>
                        <td>${f.structural_category}</td>
                        <td style="font-family:monospace;font-size:0.75rem">${f.top_geometric_feature}</td>
                        <td>${fmtVal(f.geom_pr_auc)}</td>
                        <td>${fmtVal(f.interpro_protein_f1)}</td>
                        <td>${fmtVal(f.interpro_residue_f1)}</td>
                        <td>${fmtVal(f.cath_residue_f1)}</td>
                        <td>${fmtVal(f.motif_f1)}</td>
                        <td>${fmtVal(f.composite_score)}</td>
                    </tr>
                `).join("")}
            </tbody>
        `;
        card.appendChild(table);

        // Feature importance comparison (mini bar chart per feature)
        const impDiv = document.createElement("details");
        impDiv.style.marginTop = "0.5rem";
        impDiv.innerHTML = `<summary style="font-size:0.85rem;cursor:pointer">Feature importance breakdown</summary>`;
        const impTable = document.createElement("table");
        impTable.style.fontSize = "0.75rem";
        impTable.style.width = "100%";
        impTable.style.marginTop = "0.3rem";

        // Collect all unique importance keys across features in this group
        const allKeys = new Set();
        for (const f of group.features) {
            for (const [k] of f.top_importances || []) {
                allKeys.add(k);
            }
        }
        const sortedKeys = [...allKeys].slice(0, 8);

        impTable.innerHTML = `
            <thead>
                <tr>
                    <th>Feature</th>
                    ${sortedKeys.map((k) => `<th style="font-family:monospace;font-size:0.65rem">${k.replace(/_/g, " ")}</th>`).join("")}
                </tr>
            </thead>
            <tbody>
                ${group.features.map((f) => {
                    const impMap = {};
                    for (const [k, v] of f.top_importances || []) {
                        impMap[k] = v;
                    }
                    return `<tr>
                        <td><strong>${f.feature_id}</strong></td>
                        ${sortedKeys.map((k) => {
                            const v = impMap[k] || 0;
                            const pct = Math.min(v * 100 / 0.3, 100);
                            return `<td>
                                <div style="background:linear-gradient(90deg, #4363d8 ${pct}%, transparent ${pct}%);padding:1px 3px;border-radius:2px;color:${pct > 40 ? '#fff' : '#333'}">${v > 0 ? v.toFixed(3) : ""}</div>
                            </td>`;
                        }).join("")}
                    </tr>`;
                }).join("")}
            </tbody>
        `;
        impDiv.appendChild(impTable);
        card.appendChild(impDiv);

        container.appendChild(card);
    }
}

// ── Main ──

document.addEventListener("DOMContentLoaded", async () => {
    try {
        const res = await fetch("/api/subdomain-case-study");
        if (!res.ok) {
            document.getElementById("global-stats").innerHTML =
                `<p style="color:red">Case study not built yet. Run: <code>python scripts/build_subdomain_case_study.py --data-dir feature_data_cluster</code></p>`;
            return;
        }
        const data = await res.json();

        renderGlobalStats(document.getElementById("global-stats"), data.global_stats);
        renderScatter("scatter-chart", data.groups);
        renderGroups(document.getElementById("groups-container"), data.groups);

    } catch (err) {
        console.error("Failed to load sub-domain case study:", err);
        document.getElementById("global-stats").innerHTML =
            `<p style="color:red">Error: ${err.message}</p>`;
    }
});
