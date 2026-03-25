/**
 * feature_detail.js — Orchestrates the feature detail page.
 *
 * On page load:
 *   1. Extracts feature_id from the URL path (/feature/{id})
 *   2. Fetches /api/feature/{id}, /api/feature/{id}/interpro, /api/feature/{id}/geometry in parallel
 *   3. Renders summary stat cards (Section 1)
 *   4. Renders top 5 protein entries with sequence strips + 3D viewers (Section 2)
 *   5. Renders activation bins as collapsible <details> sections (Section 3)
 *   6. Renders geometry plots if geometry data exists (Section 4)
 *
 * Dependencies (loaded before this script in feature.html):
 *   - sequence_strip.js: createSequenceStrip()
 *   - mol_viewer.js: lazyLoadViewer(), createMolViewer()
 *   - profile_plots.js: renderGeometryPlots()
 */

// ============================================================
// URL parsing
// ============================================================

/**
 * Extract the feature ID from the current URL path.
 * Expected URL format: /feature/{integer_id}
 *
 * @returns {number|null} The feature ID, or null if parsing fails.
 */
function getFeatureIdFromUrl() {
    const parts = window.location.pathname.split("/");
    // URL is /feature/{id}, so id is the last path segment
    const idStr = parts[parts.length - 1];
    const id = parseInt(idStr, 10);
    return isNaN(id) ? null : id;
}

// ============================================================
// Summary stat cards (Section 1)
// ============================================================

/**
 * Render summary statistics cards from the feature, interpro, and geometry data.
 *
 * Creates card elements for: Coverage, InterPro protein-level, InterPro residue-level,
 * Geometry protein-level, Geometry residue-level, Motif superposition.
 * Missing enrichment data (404) causes those sections to show "Pending" or similar.
 *
 * @param {HTMLElement} container     - The #summary-cards div.
 * @param {Object} featureData        - Feature JSON from /api/feature/{id}.
 * @param {Object|null} interproData  - InterPro enrichment JSON, or null if 404.
 * @param {Object|null} geometryData  - Geometry enrichment JSON, or null if 404.
 */
function renderSummaryCards(container, featureData, interproData, geometryData) {
    container.innerHTML = "";

    const cov = featureData.dataset_coverage || {};

    // --- Coverage card ---
    const coverageCard = createStatCard("Coverage", `
        <div class="value">${cov.pct_proteins_activated ?? "—"}% proteins</div>
        <div class="detail">
            ${cov.n_proteins_activated ?? "?"}/${cov.total_proteins ?? "?"} proteins,
            ${cov.n_clusters_activated ?? "?"}/${cov.total_clusters ?? "?"} clusters
            (${cov.pct_clusters_activated ?? "—"}%)
        </div>
        <div class="detail">Activation threshold: ${cov.activation_threshold ?? "—"}</div>
    `);
    container.appendChild(coverageCard);

    // --- InterPro protein-level card ---
    container.appendChild(renderInterproProteinCard(interproData));

    // --- InterPro residue-level card ---
    container.appendChild(renderInterproResidueCard(interproData));

    // --- Geometry protein-level card ---
    container.appendChild(renderGeometryProteinCard(geometryData));

    // --- Geometry residue-level card ---
    container.appendChild(renderGeometryResidueCard(geometryData));

    // --- Motif superposition card ---
    container.appendChild(renderMotifCard(geometryData));
}

/**
 * Create a stat card article element.
 *
 * @param {string} title   - Card header text.
 * @param {string} bodyHtml - Inner HTML for the card body.
 * @returns {HTMLElement}   - The article element.
 */
function createStatCard(title, bodyHtml) {
    const card = document.createElement("article");
    card.className = "stat-card";
    card.innerHTML = `<header><strong>${title}</strong></header>${bodyHtml}`;
    return card;
}

/**
 * Create a "Pending" placeholder card for enrichment sections that returned 404.
 *
 * @param {string} title  - Card header text.
 * @returns {HTMLElement}  - The article element.
 */
function pendingCard(title) {
    return createStatCard(title, '<span class="status-badge status-pending">Pending</span>');
}

/** Utility: format a number or return "—" */
function fmtVal(v, decimals = 3) {
    if (v === null || v === undefined) return "—";
    return Number(v).toFixed(decimals);
}

/**
 * Render the InterPro protein-level summary card.
 *
 * Shows the best annotation's name, F1, threshold, precision, recall,
 * TP/FP/FN counts, and interpretation string.
 *
 * @param {Object|null} data - InterPro enrichment JSON, or null if 404.
 * @returns {HTMLElement}
 */
function renderInterproProteinCard(data) {
    if (!data) return pendingCard("InterPro Protein-Level");

    const entries = data.protein_level || [];
    if (entries.length === 0) {
        return createStatCard("InterPro Protein-Level",
            '<div class="detail">No annotations tested</div>');
    }

    // Best by F1 — field is "best_f1" in pipeline output
    const best = entries.reduce((a, b) => ((b.best_f1 || 0) > (a.best_f1 || 0) ? b : a), entries[0]);

    return createStatCard("InterPro Protein-Level", `
        <div class="value">F1 = ${fmtVal(best.best_f1)}</div>
        <div class="detail"><strong>${best.annotation_name || "—"}</strong></div>
        <div class="detail">
            Threshold: ${fmtVal(best.best_threshold_normalized, 2)} (norm) / ${fmtVal(best.best_threshold, 2)} (abs)
        </div>
        <div class="detail">
            Precision: ${fmtVal(best.precision_at_best)} · Recall: ${fmtVal(best.recall_at_best)}
        </div>
        <div class="detail">
            TP: ${best.n_true_positives ?? "—"} · FP: ${best.n_false_positives ?? "—"} · FN: ${best.n_false_negatives ?? "—"}
        </div>
        ${best.interpretation ? `<div class="detail" style="margin-top:0.3rem;font-style:italic">${best.interpretation}</div>` : ""}
    `);
}

/**
 * Render the InterPro residue-level summary card.
 *
 * @param {Object|null} data - InterPro enrichment JSON, or null if 404.
 * @returns {HTMLElement}
 */
function renderInterproResidueCard(data) {
    if (!data) return pendingCard("InterPro Residue-Level");

    const entries = data.residue_level || [];
    if (entries.length === 0) {
        return createStatCard("InterPro Residue-Level",
            '<div class="detail">No annotations tested</div>');
    }

    // Best by F1 — field is "best_f1" in pipeline output
    const best = entries.reduce((a, b) => ((b.best_f1 || 0) > (a.best_f1 || 0) ? b : a), entries[0]);

    return createStatCard("InterPro Residue-Level", `
        <div class="value">F1 = ${fmtVal(best.best_f1)}</div>
        <div class="detail"><strong>${best.annotation_name || "—"}</strong></div>
        <div class="detail">Threshold: ${fmtVal(best.best_threshold, 2)} (abs) / ${fmtVal(best.best_threshold_normalized, 2)} (norm)</div>
        <div class="detail">
            Residues: ${best.n_residues_in_domain ?? "—"} in domain / ${best.n_total_residues ?? "—"} total
        </div>
        <div class="detail">
            Precision: ${fmtVal(best.precision_at_best)} · Recall: ${fmtVal(best.recall_at_best)}
        </div>
    `);
}

/**
 * Render the geometry protein-level summary card.
 *
 * Shows R2_cv, pearson_r, monomial formula, and top features with weights.
 *
 * @param {Object|null} data - Geometry enrichment JSON, or null if 404.
 * @returns {HTMLElement}
 */
function renderGeometryProteinCard(data) {
    if (!data) return pendingCard("Geometry Protein-Level");

    const geo = data.geometric_protein_level;
    if (!geo || geo.r2_cv === undefined) {
        return createStatCard("Geometry Protein-Level",
            '<div class="detail">Not enough data</div>');
    }

    // Top features as a compact list
    const topFeatures = (geo.top_features || [])
        .slice(0, 5)
        .map((f) => `${f.feature}: ${fmtVal(f.weight, 3)}`)
        .join(", ");

    return createStatCard("Geometry Protein-Level", `
        <div class="value">R2 CV = ${fmtVal(geo.r2_cv)}</div>
        <div class="detail">Pearson r: ${fmtVal(geo.pearson_r)}</div>
        <div class="detail" style="font-family:monospace;font-size:0.8rem;margin-top:0.3rem">
            ${geo.monomial || "—"}
        </div>
        ${topFeatures ? `<div class="detail" style="margin-top:0.3rem">Top: ${topFeatures}</div>` : ""}
    `);
}

/**
 * Render the geometry residue-level summary card.
 *
 * Shows GBM AUC, Tree F1, and concordance metrics.
 *
 * @param {Object|null} data - Geometry enrichment JSON, or null if 404.
 * @returns {HTMLElement}
 */
function renderGeometryResidueCard(data) {
    if (!data) return pendingCard("Geometry Residue-Level");

    const geo = data.geometric_residue_level;
    if (!geo || geo.gbm_auc_cv === undefined) {
        return createStatCard("Geometry Residue-Level",
            '<div class="detail">Not enough data</div>');
    }

    const conc = geo.concordance || {};

    return createStatCard("Geometry Residue-Level", `
        <div class="value">GBM AUC = ${fmtVal(geo.gbm_auc_cv)}</div>
        <div class="detail">Tree F1: ${fmtVal(geo.tree_f1_cv)}</div>
        <div class="detail" style="margin-top:0.3rem"><strong>Concordance:</strong></div>
        <div class="detail">
            Spearman r: ${fmtVal(conc.spearman_r)} ·
            AUROC: ${fmtVal(conc.residue_auroc)}
        </div>
        <div class="detail">
            F1: ${fmtVal(conc.f1)} ·
            Precision: ${fmtVal(conc.precision)} ·
            Recall: ${fmtVal(conc.recall)}
        </div>
    `);
}

/**
 * Render the motif superposition summary card.
 *
 * @param {Object|null} data - Geometry enrichment JSON, or null if 404.
 * @returns {HTMLElement}
 */
function renderMotifCard(data) {
    if (!data) return pendingCard("Motif Superposition");

    const motif = data.geometric_residue_level?.motif_superposition;
    if (!motif) {
        return createStatCard("Motif Superposition",
            '<div class="detail">Not computed</div>');
    }

    return createStatCard("Motif Superposition", `
        <div class="value">RMSD = ${fmtVal(motif.mean_rmsd, 2)} A</div>
        <div class="detail">
            ${motif.n_fragments ?? "?"} fragments, Std: ${fmtVal(motif.std_rmsd, 2)} A
        </div>
        <div class="detail">
            Per-position flexibility: ${(motif.per_position_flexibility || []).length} values
        </div>
    `);
}

// ============================================================
// Section 2: Top 5 Most Activating Proteins
// ============================================================

/**
 * Render the top 5 most activating protein entries, each with:
 *   - Label (accession, max activation, sequence length)
 *   - Sequence strip (canvas colored by activation)
 *   - InterPro domain overlay on the strip
 *   - 3D viewer (lazy-loaded)
 *
 * @param {HTMLElement} container    - The #top-proteins-container div.
 * @param {Object} featureData       - Feature JSON from /api/feature/{id}.
 * @param {Object|null} interproData - InterPro enrichment, for best annotation name.
 */
function renderTopProteins(container, featureData, interproData) {
    container.innerHTML = "";

    const topSeqs = (featureData.top_sequences || []).slice(0, 5);
    if (topSeqs.length === 0) {
        container.innerHTML = '<p class="secondary">No activating proteins found.</p>';
        return;
    }

    const featureMaxAct = featureData.max_activation || 1;

    // Determine best interpro annotation name for domain overlay highlighting
    const bestAnnotationName = getBestAnnotationName(interproData);

    for (const protein of topSeqs) {
        const entry = createProteinEntry(protein, featureMaxAct, bestAnnotationName);
        container.appendChild(entry);
    }
}

/**
 * Extract the best protein-level annotation name from interpro data.
 *
 * @param {Object|null} interproData - InterPro enrichment data.
 * @returns {string|null} Best annotation name, or null.
 */
function getBestAnnotationName(interproData) {
    if (!interproData) return null;
    const entries = interproData.protein_level || [];
    if (entries.length === 0) return null;
    const best = entries.reduce((a, b) => ((b.best_f1 || 0) > (a.best_f1 || 0) ? b : a), entries[0]);
    return best.annotation_name || null;
}

/**
 * Create a DOM element for a single protein entry with sequence strip + 3D viewer.
 *
 * @param {Object} protein          - Protein data from top_sequences or activation_bins.
 * @param {number} featureMaxAct    - Feature-level max activation (for color normalization).
 * @param {string|null} bestAnnotationName - InterPro annotation to highlight.
 * @returns {HTMLElement}
 */
function createProteinEntry(protein, featureMaxAct, bestAnnotationName) {
    const entry = document.createElement("div");
    entry.className = "protein-entry";

    // Label
    const label = document.createElement("div");
    label.className = "protein-label";
    label.textContent = `${protein.accession} · max: ${fmtVal(protein.max_activation, 4)} · ${protein.sequence_length || protein.sequence?.length || "?"} residues`;
    entry.appendChild(label);

    // Sequence strip (canvas with activation coloring + domain overlay)
    const stripDiv = document.createElement("div");
    entry.appendChild(stripDiv);
    createSequenceStrip(stripDiv, {
        sequence: protein.sequence || "",
        activations: protein.per_residue_activations || [],
        maxActivation: featureMaxAct,
        accession: protein.accession,
        bestAnnotationName: bestAnnotationName,
    });

    // 3D viewer (lazy-loaded)
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
    } else {
        const placeholder = document.createElement("div");
        placeholder.className = "viewer-container";
        placeholder.innerHTML = '<div class="viewer-placeholder">No structure available</div>';
        entry.appendChild(placeholder);
    }

    return entry;
}

// ============================================================
// Section 3: Activation Bins (collapsible)
// ============================================================

/**
 * Render collapsible activation bin sections.
 *
 * Creates a <details> element for each bin (0.75-1.0, 0.5-0.75, etc.).
 * Proteins within each bin get sequence strips + 3D viewers, but PDB loading
 * is deferred until the <details> is opened (lazy).
 *
 * @param {HTMLElement} container    - The #bins-container div.
 * @param {Object} featureData       - Feature JSON with activation_bins dict.
 * @param {number} featureMaxAct     - Feature-level max activation.
 * @param {string|null} bestAnnotationName - InterPro annotation for domain overlay.
 */
function renderBins(container, featureData, featureMaxAct, bestAnnotationName) {
    container.innerHTML = "";

    const bins = featureData.activation_bins || {};
    // Display bins in descending order (highest activation first)
    const binKeys = ["0.75-1.0", "0.5-0.75", "0.25-0.5", "0.0-0.25"];

    for (const binKey of binKeys) {
        const proteins = bins[binKey] || [];

        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = `${binKey} (${proteins.length} proteins)`;
        details.appendChild(summary);

        // Inner content div (populated on first open)
        const contentDiv = document.createElement("div");
        contentDiv.dataset.loaded = "false";
        details.appendChild(contentDiv);

        // Lazy: only render protein entries when <details> is opened
        details.addEventListener("toggle", () => {
            if (details.open && contentDiv.dataset.loaded === "false") {
                contentDiv.dataset.loaded = "true";
                if (proteins.length === 0) {
                    contentDiv.innerHTML = '<p class="secondary">No proteins in this bin.</p>';
                    return;
                }
                for (const protein of proteins) {
                    const entry = createProteinEntry(protein, featureMaxAct, bestAnnotationName);
                    contentDiv.appendChild(entry);
                }
            }
        });

        container.appendChild(details);
    }
}

// ============================================================
// Main: fetch data and orchestrate rendering
// ============================================================

document.addEventListener("DOMContentLoaded", async () => {
    const featureId = getFeatureIdFromUrl();
    if (featureId === null) {
        document.getElementById("page-title").textContent = "Error: Invalid feature ID";
        return;
    }

    document.getElementById("page-title").textContent = `Feature ${featureId}`;
    document.title = `Feature ${featureId} — SAE Visualizer`;

    try {
        // Fetch all three endpoints in parallel
        // InterPro and Geometry may 404 — that's expected for missing enrichment
        const [featureRes, interproRes, geometryRes] = await Promise.all([
            fetch(`/api/feature/${featureId}`),
            fetch(`/api/feature/${featureId}/interpro`).catch(() => null),
            fetch(`/api/feature/${featureId}/geometry`).catch(() => null),
        ]);

        if (!featureRes.ok) {
            throw new Error(`Feature fetch failed: ${featureRes.status}`);
        }

        const featureData = await featureRes.json();
        const interproData = interproRes && interproRes.ok ? await interproRes.json() : null;
        const geometryData = geometryRes && geometryRes.ok ? await geometryRes.json() : null;

        const featureMaxAct = featureData.max_activation || 1;
        const bestAnnotationName = getBestAnnotationName(interproData);

        // Section 1: Summary stats
        renderSummaryCards(
            document.getElementById("summary-cards"),
            featureData, interproData, geometryData
        );

        // Section 2: Top 5 proteins
        renderTopProteins(
            document.getElementById("top-proteins-container"),
            featureData, interproData
        );

        // Section 3: Activation bins
        renderBins(
            document.getElementById("bins-container"),
            featureData, featureMaxAct, bestAnnotationName
        );

        // Section 4: Geometry plots (only if data exists)
        if (geometryData) {
            document.getElementById("geometry-section").style.display = "";
            renderGeometryPlots(
                document.getElementById("geometry-container"),
                geometryData
            );
        }
    } catch (err) {
        console.error("Failed to load feature data:", err);
        document.getElementById("summary-cards").innerHTML =
            `<p style="color:red">Error loading feature: ${err.message}</p>`;
    }
});
