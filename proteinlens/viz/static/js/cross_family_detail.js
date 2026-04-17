/**
 * cross_family_detail.js — Deep dive page for a single geometry-primary feature.
 *
 * URL: /cross-family-geometry/{feature_id}
 *
 * Shows:
 *   1. Summary stat cards (geometry + interpro + motif)
 *   2. InterPro family breakdown (bar chart + table)
 *   3. Geometry feature importance (horizontal bar)
 *   4. Motif superposition (3D viewer + flexibility chart)
 *   5. Per-family protein examples (top protein from each family's TP set)
 *   6. Geometry enrichment plots (dual-axis, concordance)
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

function getFeatureIdFromUrl() {
    const parts = window.location.pathname.split("/");
    const id = parseInt(parts[parts.length - 1], 10);
    return isNaN(id) ? null : id;
}

// ============================================================
// 1. Summary Cards
// ============================================================

function renderSummaryCards(container, feat) {
    container.innerHTML = "";

    container.appendChild(createStatCard("Geometry Composite", `
        <div class="value">${fmtVal(feat.composite_score)}</div>
        <div class="detail">${feat.structural_category}</div>
        <div class="detail" style="font-family:monospace">${feat.top_geometric_feature}</div>
    `));

    container.appendChild(createStatCard("Geometry Classifiers", `
        <div class="value">GBM AUC = ${fmtVal(feat.gbm_auc_cv)}</div>
        <div class="detail">Tree F1: ${fmtVal(feat.tree_f1_cv)}</div>
        <div class="detail">Concordance F1: ${fmtVal(feat.concordance_f1)}</div>
        <div class="detail">PR-AUC: ${fmtVal(feat.concordance_prauc)}</div>
    `));

    container.appendChild(createStatCard("InterPro Protein-Level", `
        <div class="value">F1 = ${fmtVal(feat.best_interpro_protein_f1)}</div>
        <div class="detail">${feat.best_interpro_protein_name}</div>
        <div class="detail">${feat.n_families_above_03} families with F1 &gt; 0.3</div>
        <div class="detail">${feat.n_families_above_05} families with F1 &gt; 0.5</div>
    `));

    container.appendChild(createStatCard("InterPro Residue-Level", `
        <div class="value">F1 = ${fmtVal(feat.best_interpro_residue_f1)}</div>
        <div class="detail">${feat.best_interpro_residue_name || "\u2014"}</div>
        <div class="detail" style="color:#dc2626">Below null threshold \u2014 family fails at residue level</div>
    `));

    container.appendChild(createStatCard("Motif Superposition", `
        <div class="value">RMSD/pos = ${fmtVal(feat.motif_rmsd_per_pos)} \u00c5</div>
        <div class="detail">Raw: ${fmtVal(feat.motif_rmsd, 2)} \u00c5 over ${feat.motif_length} positions</div>
        <div class="detail">${feat.motif_n_fragments} fragments, Std: ${fmtVal(feat.motif_std_rmsd, 2)} \u00c5</div>
    `));

    container.appendChild(createStatCard("Sequence Metrics", `
        <div class="detail">Motif PR-AUC: ${fmtVal(feat.motif_seq_pr_auc)} ${feat.motif_seq_best ? `(<span style="font-family:monospace">${feat.motif_seq_best}</span>)` : ""}</div>
        <div class="detail">Position F1: ${fmtVal(feat.position_f1)}</div>
        <div class="detail" style="color:#16a34a">Both below null \u2014 sequence cannot explain</div>
    `));
}

// ============================================================
// 2. InterPro Family Breakdown
// ============================================================

function renderInterproBreakdown(chartDiv, tableDiv, families) {
    // Bar chart: F1, Precision, Recall per family
    const names = families.map(f => f.name.length > 40 ? f.name.slice(0, 37) + "..." : f.name);
    const fullNames = families.map(f => f.name);

    const traces = [
        {
            y: names, x: families.map(f => f.f1), name: "F1",
            type: "bar", orientation: "h", marker: { color: "#2563eb" },
            text: fullNames, hovertemplate: "%{text}<br>F1: %{x:.3f}<extra></extra>",
        },
        {
            y: names, x: families.map(f => f.precision), name: "Precision",
            type: "bar", orientation: "h", marker: { color: "#16a34a" },
            text: fullNames, hovertemplate: "%{text}<br>Precision: %{x:.3f}<extra></extra>",
        },
        {
            y: names, x: families.map(f => f.recall), name: "Recall",
            type: "bar", orientation: "h", marker: { color: "#d97706" },
            text: fullNames, hovertemplate: "%{text}<br>Recall: %{x:.3f}<extra></extra>",
        },
    ];

    const layout = {
        barmode: "group",
        xaxis: { title: "Score", range: [0, 1.05] },
        yaxis: { autorange: "reversed", tickfont: { size: 10 } },
        height: Math.max(250, families.length * 50 + 80),
        margin: { t: 10, b: 40, l: 280, r: 20 },
        legend: { x: 0.7, y: 0.99 },
        shapes: [{
            type: "line", xref: "x", yref: "paper",
            x0: 0.3, x1: 0.3, y0: 0, y1: 1,
            line: { color: "#999", dash: "dash", width: 1 },
        }],
    };

    Plotly.newPlot(chartDiv, traces, layout, { responsive: true, displayModeBar: false });

    // Table
    const rows = families.map(f => `<tr>
        <td style="font-size:0.85rem">${f.name}</td>
        <td style="font-family:monospace;font-size:0.8rem">${f.code}</td>
        <td>${fmtVal(f.f1)}</td>
        <td>${fmtVal(f.precision)}</td>
        <td>${fmtVal(f.recall)}</td>
        <td>${f.tp}</td><td>${f.fp}</td><td>${f.fn}</td>
    </tr>`).join("");

    tableDiv.innerHTML = `
        <table role="grid" style="font-size:0.85rem;">
            <thead><tr>
                <th>Family</th><th>Code</th><th>F1</th><th>Prec</th><th>Rec</th>
                <th>TP</th><th>FP</th><th>FN</th>
            </tr></thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}

// ============================================================
// 3. Geometry Feature Importance
// ============================================================

function renderImportanceChart(div, importances) {
    // Sort and take top 15
    const sorted = Object.entries(importances)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 15);

    const names = sorted.map(([k]) => k);
    const vals = sorted.map(([, v]) => v);

    const traces = [{
        y: names.reverse(),
        x: vals.reverse(),
        type: "bar",
        orientation: "h",
        marker: {
            color: vals.map(v => {
                const norm = v / Math.max(...vals, 0.01);
                const r = Math.round(norm * 220 + 35);
                return `rgb(${r}, ${Math.round(38 + (1 - norm) * 100)}, ${Math.round(38 + (1 - norm) * 100)})`;
            }),
        },
        hovertemplate: "%{y}: %{x:.4f}<extra></extra>",
    }];

    const layout = {
        xaxis: { title: "Importance" },
        height: Math.max(250, sorted.length * 25 + 60),
        margin: { t: 10, b: 40, l: 200, r: 20 },
    };

    Plotly.newPlot(div, traces, layout, { responsive: true, displayModeBar: false });
}

// ============================================================
// 4. Motif Superposition (reuses mol_viewer.js)
// ============================================================

function renderMotifSection(viewerDiv, flexDiv, statsDiv, geometryData) {
    const motif = geometryData.geometric_residue_level?.motif_superposition;
    if (!motif || !motif.mean_structure_pdb) {
        statsDiv.textContent = "No motif superposition data.";
        return;
    }

    const motifLen = (motif.per_position_flexibility || []).length;
    const normRmsd = motifLen > 0 ? (motif.mean_rmsd / motifLen).toFixed(3) : "\u2014";

    statsDiv.textContent = `RMSD/pos: ${normRmsd} \u00c5 ` +
        `(raw: ${motif.mean_rmsd?.toFixed(2) ?? "\u2014"} \u00c5 over ${motifLen} positions), ` +
        `${motif.n_fragments ?? "?"} fragments, ` +
        `Std: ${motif.std_rmsd?.toFixed(2) ?? "\u2014"} \u00c5`;

    const flexibility = motif.per_position_flexibility || [];
    createMotifViewer(viewerDiv, motif.mean_structure_pdb, flexibility);

    if (flexibility.length > 0) {
        renderFlexibilityChart(flexDiv, flexibility);
    }
}

// ============================================================
// 5. Per-Family Protein Examples
// ============================================================

function renderPerFamilyProteins(container, feat, featureData, interproData) {
    container.innerHTML = "";

    const topSeqs = featureData.top_sequences || [];
    if (topSeqs.length === 0) {
        container.innerHTML = '<p class="secondary">No activating proteins available.</p>';
        return;
    }

    const featureMaxAct = featureData.max_activation || 1;
    const families = feat.interpro_families.filter(f => f.f1 > 0.3);

    // For each family, find proteins in top sequences that are TPs for that family
    // We don't have per-protein family labels in the feature JSON, so we show
    // top proteins with the family context as a reference
    // Group by showing each family as a section header, then the top proteins

    // First show the top 5 proteins with full rendering
    const topSection = document.createElement("div");
    topSection.style.marginBottom = "1.5rem";
    const topHeading = document.createElement("h3");
    topHeading.textContent = "Top Activating Proteins";
    topSection.appendChild(topHeading);

    const topNote = document.createElement("p");
    topNote.className = "secondary";
    topNote.style.fontSize = "0.85rem";
    topNote.textContent = "These proteins come from multiple unrelated families but share the same geometric motif at their activation sites.";
    topSection.appendChild(topNote);

    for (const protein of topSeqs.slice(0, 5)) {
        const entry = document.createElement("div");
        entry.className = "protein-entry";

        const label = document.createElement("div");
        label.className = "protein-label";
        label.textContent = `${protein.accession} \u00b7 max: ${fmtVal(protein.max_activation, 4)} \u00b7 ${protein.sequence?.length || "?"} residues`;
        entry.appendChild(label);

        createTextSequence(entry, {
            sequence: protein.sequence || "",
            activations: protein.per_residue_activations || [],
            maxActivation: featureMaxAct,
            accession: protein.accession,
            maxAct: protein.max_activation,
            showLabel: false,
        });

        const stripDiv = document.createElement("div");
        entry.appendChild(stripDiv);

        // Get best annotation name from interpro data
        let bestAnnotationName = null;
        if (interproData) {
            const resEntries = interproData.residue_level || [];
            if (resEntries.length > 0) {
                bestAnnotationName = resEntries.reduce((a, b) =>
                    ((b.best_f1 || 0) > (a.best_f1 || 0) ? b : a), resEntries[0]
                ).annotation_name || null;
            }
        }

        createSequenceStrip(stripDiv, {
            sequence: protein.sequence || "",
            activations: protein.per_residue_activations || [],
            maxActivation: featureMaxAct,
            accession: protein.accession,
            bestAnnotationName: bestAnnotationName,
        });

        // 3D viewer
        if (protein.pdb_available !== false) {
            const viewerDiv = document.createElement("div");
            viewerDiv.className = "viewer-container";
            entry.appendChild(viewerDiv);
            lazyLoadViewer(
                viewerDiv,
                protein.accession,
                protein.per_residue_activations || [],
                featureMaxAct
            );
        }

        topSection.appendChild(entry);
    }

    container.appendChild(topSection);

    // Show the family context table
    if (families.length > 0) {
        const familyNote = document.createElement("div");
        familyNote.style.marginTop = "1.5rem";
        familyNote.innerHTML = `
            <h3>InterPro Family Context</h3>
            <p class="secondary" style="font-size:0.85rem">
                The feature fires on proteins from ${families.length} different families (F1 &gt; 0.3).
                If this were a sequence-level feature, we'd expect one dominant family with high F1.
                Instead, the partial match across many families shows the <strong>geometric motif</strong>
                is the invariant.
            </p>
            <table role="grid" style="font-size:0.85rem;">
                <thead><tr><th>Family</th><th>F1</th><th>Precision</th><th>Recall</th><th>TP</th><th>FP</th></tr></thead>
                <tbody>${families.map(f => `<tr>
                    <td>${f.name}</td>
                    <td>${fmtVal(f.f1)}</td>
                    <td>${fmtVal(f.precision)}</td>
                    <td>${fmtVal(f.recall)}</td>
                    <td>${f.tp}</td>
                    <td>${f.fp}</td>
                </tr>`).join("")}</tbody>
            </table>
        `;
        container.appendChild(familyNote);
    }
}

// ============================================================
// 6. Geometry plots (reuses profile_plots.js)
// ============================================================

function renderGeometrySection(container, geometryData) {
    const plotData = geometryData.plot_data;
    if (!plotData || !plotData.top_proteins || plotData.top_proteins.length === 0) {
        return;
    }

    document.getElementById("geometry-plots-section").style.display = "";

    for (const protein of plotData.top_proteins.slice(0, 3)) {
        const section = document.createElement("div");
        section.className = "protein-entry";

        const label = document.createElement("div");
        label.className = "protein-label";
        label.textContent = `${protein.accession} (${protein.sequence?.length || "?"} residues)`;
        section.appendChild(label);

        const overlayDiv = document.createElement("div");
        overlayDiv.className = "plot-container";
        section.appendChild(overlayDiv);
        renderDualAxisPlot(overlayDiv, protein);

        if (protein.concordance_labels && protein.concordance_labels.length > 0) {
            const concordDiv = document.createElement("div");
            concordDiv.style.marginBottom = "0.75rem";
            section.appendChild(concordDiv);
            renderConcordanceStrip(concordDiv, protein.concordance_labels);
        }

        container.appendChild(section);
    }
}

// ============================================================
// Main
// ============================================================

document.addEventListener("DOMContentLoaded", async () => {
    const featureId = getFeatureIdFromUrl();
    if (featureId === null) {
        document.getElementById("page-title").textContent = "Error: Invalid feature ID";
        return;
    }

    document.getElementById("page-title").textContent = `Feature ${featureId} — Cross-Family Geometry`;
    document.title = `Feature ${featureId} — Cross-Family Geometry`;

    try {
        // Fetch all data in parallel
        const [cfRes, featureRes, interproRes, geometryRes] = await Promise.all([
            fetch("/api/cross-family-geometry"),
            fetch(`/api/feature/${featureId}`),
            fetch(`/api/feature/${featureId}/interpro`).catch(() => null),
            fetch(`/api/feature/${featureId}/geometry`).catch(() => null),
        ]);

        if (!cfRes.ok) throw new Error("Cross-family data not found");
        if (!featureRes.ok) throw new Error(`Feature ${featureId} not found`);

        const cfData = await cfRes.json();
        const featureData = await featureRes.json();
        const interproData = interproRes && interproRes.ok ? await interproRes.json() : null;
        const geometryData = geometryRes && geometryRes.ok ? await geometryRes.json() : null;

        // Find this feature in the cross-family data
        const feat = cfData.features.find(f => f.feature_id === featureId);
        if (!feat) throw new Error(`Feature ${featureId} not in cross-family geometry data`);

        // 1. Summary cards
        renderSummaryCards(document.getElementById("summary-cards"), feat);

        // 2. InterPro breakdown
        renderInterproBreakdown(
            document.getElementById("interpro-chart"),
            document.getElementById("interpro-table"),
            feat.interpro_families
        );

        // 3. Feature importance + radar glyph
        renderImportanceChart(
            document.getElementById("importance-chart"),
            feat.feature_importances
        );
        if (feat.feature_importances && typeof aggregateToCategories === "function") {
            const scores = aggregateToCategories(feat.feature_importances);
            if (scores) {
                const radarDiv = document.createElement("div");
                radarDiv.style.cssText = "margin:1rem auto; text-align:center;";
                const chartEl = document.getElementById("importance-chart");
                chartEl.parentNode.insertBefore(radarDiv, chartEl);
                renderRadarWithLegend(radarDiv, scores, { size: 200 });
            }
        }

        // 4. Motif superposition
        if (geometryData) {
            renderMotifSection(
                document.getElementById("motif-viewer"),
                document.getElementById("flexibility-chart"),
                document.getElementById("motif-stats"),
                geometryData
            );
        }

        // 5. Per-family proteins
        renderPerFamilyProteins(
            document.getElementById("per-family-container"),
            feat, featureData, interproData
        );

        // 6. Geometry plots
        if (geometryData) {
            renderGeometrySection(
                document.getElementById("geometry-plots-container"),
                geometryData
            );
        }

    } catch (err) {
        console.error("Failed to load:", err);
        document.getElementById("summary-cards").innerHTML =
            `<p style="color:red">Error: ${err.message}</p>`;
    }
});
