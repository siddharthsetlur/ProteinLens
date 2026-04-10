/**
 * feature_detail.js — Orchestrates the feature detail page.
 *
 * On page load:
 *   1. Extracts feature_id from the URL path (/feature/{id})
 *   2. Fetches /api/feature/{id}, /api/feature/{id}/interpro, /api/feature/{id}/geometry, /api/feature/{id}/motif, /api/feature/{id}/position in parallel
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
 * @param {Object|null} motifData    - Motif enrichment JSON, or null if 404.
 * @param {Object|null} positionData - Position enrichment JSON, or null if 404.
 * @param {Object|null} gpInfo       - Geometry-primary info for this feature, or null.
 */
function renderSummaryCards(container, featureData, interproData, geometryData, motifData, positionData, cathData, gpInfo) {
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
    container.appendChild(renderGeometryResidueCard(geometryData, gpInfo));

    // --- Sequence motif F1 card ---
    container.appendChild(renderSequenceMotifCard(motifData));

    // --- Position F1 card ---
    container.appendChild(renderPositionCard(positionData));

    // --- CATH F1 card ---
    container.appendChild(renderCathCard(cathData));

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
function renderGeometryResidueCard(data, gpInfo) {
    if (!data) return pendingCard("Geometry Residue-Level");

    const geo = data.geometric_residue_level;
    if (!geo || geo.gbm_auc_cv === undefined) {
        return createStatCard("Geometry Residue-Level",
            '<div class="detail">Not enough data</div>');
    }

    const conc = geo.concordance || {};

    let gpBadge = "";
    if (gpInfo && gpInfo.is_geometry_primary) {
        gpBadge = `
            <div style="background:#fff3cd;border:1px solid #ffc107;border-radius:4px;padding:0.4rem 0.6rem;margin-bottom:0.5rem;">
                <strong style="color:#856404;">Geometry-primary</strong>
                <span style="color:#856404;"> (score: ${fmtVal(gpInfo.composite_score)})</span>
                <div class="detail" style="color:#856404;margin-top:0.2rem;">
                    Activation best explained by local 3D structure.
                    Category: <strong>${gpInfo.structural_category || "\u2014"}</strong>
                    (${gpInfo.top_geometric_feature || "\u2014"})
                </div>
                <div class="detail" style="color:#856404;">
                    All sequence metrics below null p95:
                    Motif F1=${fmtVal(gpInfo.motif_f1)} \u2264 0.71,
                    Position F1=${fmtVal(gpInfo.position_f1)} \u2264 0.12,
                    InterPro Res F1=${fmtVal(gpInfo.interpro_res_f1)} \u2264 0.20
                </div>
            </div>`;
    }

    return createStatCard("Geometry Residue-Level", `
        ${gpBadge}
        <div class="value">PR-AUC = ${fmtVal(conc.avg_precision)}</div>
        <div class="detail">GBM ROC-AUC: ${fmtVal(geo.gbm_auc_cv)} · Tree F1: ${fmtVal(geo.tree_f1_cv)}</div>
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

    const motifLen = (motif.per_position_flexibility || []).length;
    const normRmsd = motifLen > 0 ? motif.mean_rmsd / motifLen : null;

    return createStatCard("Motif Superposition", `
        <div class="value">RMSD/pos = ${fmtVal(normRmsd, 3)} \u00c5</div>
        <div class="detail">
            Raw RMSD: ${fmtVal(motif.mean_rmsd, 2)} \u00c5 over ${motifLen} positions
        </div>
        <div class="detail">
            ${motif.n_fragments ?? "?"} fragments, Std: ${fmtVal(motif.std_rmsd, 2)} \u00c5
        </div>
    `);
}

/**
 * Render the sequence motif F1 enrichment card.
 *
 * Shows the best k-mer motif, its F1, threshold, precision, recall,
 * occurrence count, and interpretation string. Lists up to 5 top motifs.
 *
 * @param {Object|null} data - Motif enrichment JSON from /api/feature/{id}/motif, or null if 404.
 * @returns {HTMLElement}
 */
function renderSequenceMotifCard(data) {
    if (!data) return pendingCard("Sequence Motif F1");

    const motifs = data.top_motifs || [];
    if (motifs.length === 0) {
        return createStatCard("Sequence Motif F1",
            '<div class="detail">No eligible motifs found</div>');
    }

    const best = motifs[0];
    const otherMotifs = motifs.slice(1, 5)
        .map((m) => `<span style="font-family:monospace">${m.motif}</span> F1=${fmtVal(m.best_f1)}`)
        .join(", ");

    return createStatCard("Sequence Motif F1", `
        <div class="value">F1 = ${fmtVal(best.best_f1)}</div>
        <div class="detail"><strong style="font-family:monospace;font-size:1.1rem">${best.motif}</strong></div>
        <div class="detail">
            Threshold: ${fmtVal(best.best_threshold_normalized, 2)} (norm) / ${fmtVal(best.best_threshold, 2)} (abs)
        </div>
        <div class="detail">
            Precision: ${fmtVal(best.precision_at_best)} · Recall: ${fmtVal(best.recall_at_best)}
        </div>
        <div class="detail">
            Occurrences: ${best.n_occurrences ?? "—"} ·
            TP: ${best.n_true_positives ?? "—"} · FP: ${best.n_false_positives ?? "—"} · FN: ${best.n_false_negatives ?? "—"}
        </div>
        ${best.interpretation ? `<div class="detail" style="margin-top:0.3rem;font-style:italic">${best.interpretation}</div>` : ""}
        ${otherMotifs ? `<div class="detail" style="margin-top:0.3rem"><strong>Other top motifs:</strong> ${otherMotifs}</div>` : ""}
        <div class="detail" style="margin-top:0.3rem">
            ${data.n_proteins_evaluated ?? "?"} proteins · ${data.n_total_residues ?? "?"} residues · ${data.n_unique_kmers_tested ?? "?"} k-mers tested
        </div>
    `);
}

/**
 * Render the sequence position F1 enrichment card.
 *
 * Shows the best position predicate, its F1, threshold, precision, recall,
 * and occurrence count. Lists up to 5 top predicates.
 *
 * @param {Object|null} data - Position enrichment JSON from /api/feature/{id}/position, or null if 404.
 * @returns {HTMLElement}
 */
function renderPositionCard(data) {
    if (!data) return pendingCard("Sequence Position F1");

    const positions = data.top_positions || [];
    if (positions.length === 0) {
        return createStatCard("Sequence Position F1",
            '<div class="detail">No eligible position predicates found</div>');
    }

    const best = positions[0];
    const otherPositions = positions.slice(1, 5)
        .map((p) => `<span style="font-family:monospace">${p.position}</span> F1=${fmtVal(p.best_f1)}`)
        .join(", ");

    return createStatCard("Sequence Position F1", `
        <div class="value">F1 = ${fmtVal(best.best_f1)}</div>
        <div class="detail"><strong style="font-family:monospace;font-size:1.1rem">${best.position}</strong></div>
        <div class="detail">
            Threshold: ${fmtVal(best.best_threshold_normalized, 2)} (norm) / ${fmtVal(best.best_threshold, 2)} (abs)
        </div>
        <div class="detail">
            Precision: ${fmtVal(best.precision_at_best)} · Recall: ${fmtVal(best.recall_at_best)}
        </div>
        <div class="detail">
            Matching residues: ${best.n_occurrences ?? "—"} ·
            TP: ${best.n_true_positives ?? "—"} · FP: ${best.n_false_positives ?? "—"} · FN: ${best.n_false_negatives ?? "—"}
        </div>
        ${best.interpretation ? `<div class="detail" style="margin-top:0.3rem;font-style:italic">${best.interpretation}</div>` : ""}
        ${otherPositions ? `<div class="detail" style="margin-top:0.3rem"><strong>Other top predicates:</strong> ${otherPositions}</div>` : ""}
        <div class="detail" style="margin-top:0.3rem">
            ${data.n_proteins_evaluated ?? "?"} proteins · ${data.n_total_residues ?? "?"} residues · ${data.n_predicates_tested ?? "?"} predicates tested
        </div>
    `);
}

/**
 * Render the CATH enrichment card with per-hierarchy-level breakdown.
 *
 * Shows the best (max) residue F1 across all hierarchy levels as the headline,
 * then a table with each level's best label and residue F1.
 *
 * @param {Object|null} data - CATH enrichment JSON from /api/feature/{id}/cath, or null if 404.
 * @returns {HTMLElement}
 */
function renderCathCard(data) {
    if (!data) return pendingCard("CATH F1");

    const summary = data.summary || {};
    const levels = ["C", "CA", "CAT", "CATH"];
    const levelNames = { C: "Class", CA: "Architecture", CAT: "Topology", CATH: "Homology" };

    // Find the best residue F1 across all levels
    let bestF1 = 0;
    let bestLevel = "";
    for (const lvl of levels) {
        const f1 = (summary[lvl] || {}).top_residue_f1 || 0;
        if (f1 > bestF1) {
            bestF1 = f1;
            bestLevel = lvl;
        }
    }

    if (bestF1 === 0) {
        return createStatCard("CATH F1",
            '<div class="detail">No CATH enrichment found</div>');
    }

    // Build per-level breakdown rows
    const levelRows = levels.map((lvl) => {
        const s = summary[lvl] || {};
        const f1 = s.top_residue_f1 || 0;
        const label = s.top_residue_label || "—";
        const marker = lvl === bestLevel ? " <strong>&larr; best</strong>" : "";
        return `<tr><td>${lvl} (${levelNames[lvl]})</td><td style="font-family:monospace">${label}</td><td>${fmtVal(f1)}</td><td>${marker}</td></tr>`;
    }).join("");

    // Protein-level best F1 (for context)
    let proteinBestF1 = 0;
    let proteinBestLabel = "";
    for (const lvl of levels) {
        const pf1 = (summary[lvl] || {}).top_protein_f1 || 0;
        if (pf1 > proteinBestF1) {
            proteinBestF1 = pf1;
            proteinBestLabel = (summary[lvl] || {}).top_protein_label || "";
        }
    }

    // Find detail metrics for the best residue level
    const bestLevelResidues = (data.residue_level || {})[bestLevel] || [];
    const bestEntry = bestLevelResidues.length > 0 ? bestLevelResidues[0] : null;
    const detailHtml = bestEntry ? `
        <div class="detail">
            Precision: ${fmtVal(bestEntry.precision_at_best)} · Recall: ${fmtVal(bestEntry.recall_at_best)}
        </div>
        <div class="detail">
            TP: ${bestEntry.n_true_positives ?? "—"} · FP: ${bestEntry.n_false_positives ?? "—"} · FN: ${bestEntry.n_false_negatives ?? "—"}
        </div>
    ` : "";

    return createStatCard("CATH F1", `
        <div class="value">Residue F1 = ${fmtVal(bestF1)}</div>
        <table style="width:100%;font-size:0.85rem;margin:0.3rem 0">
            <thead><tr><th>Level</th><th>Label</th><th>Res. F1</th><th></th></tr></thead>
            <tbody>${levelRows}</tbody>
        </table>
        ${detailHtml}
        <div class="detail" style="margin-top:0.3rem">
            Protein-level best F1: ${fmtVal(proteinBestF1)} (<span style="font-family:monospace">${proteinBestLabel}</span>)
        </div>
        <div class="detail">
            ${data.n_proteins_evaluated ?? "?"} proteins evaluated · ${data.n_proteins_with_cath ?? "?"} with CATH hits
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
    const bestAnnotationName = getBestAnnotationName(interproData);

    for (const protein of topSeqs) {
        const entry = createProteinEntry(protein, featureMaxAct, bestAnnotationName);
        container.appendChild(entry);
    }
}

/**
 * Render the top 3 sequence alignment view.
 *
 * Shows a compact MSA-style block where each row is one protein's sequence
 * with per-residue activation coloring. Placed between summary cards and
 * the detailed per-protein entries.
 *
 * @param {HTMLElement} container - The #alignment-container div.
 * @param {Object} featureData   - Feature JSON from /api/feature/{id}.
 */
function renderAlignment(container, featureData) {
    container.innerHTML = "";

    const topSeqs = featureData.top_sequences || [];
    if (topSeqs.length === 0) {
        container.innerHTML = '<p class="secondary">No activating sequences.</p>';
        return;
    }

    const featureMaxAct = featureData.max_activation || 1;
    createAlignmentView(container, topSeqs, featureMaxAct, 3);
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

    // Text sequence (AA letters with activation-colored backgrounds)
    createTextSequence(entry, {
        sequence: protein.sequence || "",
        activations: protein.per_residue_activations || [],
        maxActivation: featureMaxAct,
        accession: protein.accession,
        maxAct: protein.max_activation,
        showLabel: false,  // label is already in protein-label div above
    });

    // Canvas strip (compact heatmap with InterPro domain overlay)
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
// Section 4: NMPFams Novel Metagenomic Hits
// ============================================================

/**
 * Render NMPFams protein hits with sequence strips and 3D viewers.
 *
 * @param {HTMLElement} container - The #nmpfam-container div.
 * @param {Object} nmpfamData    - NMPFams enrichment JSON from /api/feature/{id}/nmpfam.
 * @param {number} featureMaxAct - Feature-level max activation for color normalization.
 */
function renderNmpfamHits(container, nmpfamData, featureMaxAct) {
    container.innerHTML = "";

    const hits = nmpfamData.nmpfam_hits || [];
    const threshold = nmpfamData.activation_threshold || 0;

    // Summary line
    const summary = document.createElement("p");
    summary.innerHTML = `<strong>${hits.length}</strong> novel metagenomic famil${hits.length === 1 ? "y" : "ies"} activated above threshold (${fmtVal(threshold, 3)})`;
    container.appendChild(summary);

    for (const hit of hits) {
        const entry = document.createElement("div");
        entry.className = "protein-entry";

        // Label with family info and link
        const label = document.createElement("div");
        label.className = "protein-label";
        label.innerHTML = `<a href="${hit.nmpfams_url}" target="_blank">${hit.family_id}</a> · `
            + `max: ${fmtVal(hit.max_activation, 4)} (${fmtVal(hit.normalized_activation * 100, 1)}% of global max) · `
            + `${hit.sequence_length || "?"} residues · `
            + `<span style="opacity:0.7">${hit.category} · ${hit.sequence_count} members</span>`;
        entry.appendChild(label);

        // Sequence strip with activation coloring
        if (hit.per_residue_activations && hit.sequence) {
            // Text sequence
            createTextSequence(entry, {
                sequence: hit.sequence,
                activations: hit.per_residue_activations,
                maxActivation: featureMaxAct,
                accession: hit.family_id,
                maxAct: hit.max_activation,
                showLabel: false,
            });

            // Canvas strip
            const stripDiv = document.createElement("div");
            entry.appendChild(stripDiv);
            createSequenceStrip(stripDiv, {
                sequence: hit.sequence,
                activations: hit.per_residue_activations,
                maxActivation: featureMaxAct,
                accession: hit.family_id,
            });
        }

        // 3D viewer from NMPFams PDB proxy
        if (hit.pdb_available) {
            const viewerDiv = document.createElement("div");
            viewerDiv.className = "viewer-container";
            entry.appendChild(viewerDiv);
            lazyLoadNmpfamViewer(
                viewerDiv, hit.family_id,
                hit.per_residue_activations || [],
                featureMaxAct
            );
        }

        container.appendChild(entry);
    }
}

/**
 * Lazy-load a 3D viewer for an NMPFams protein, fetching the PDB via our proxy.
 */
function lazyLoadNmpfamViewer(container, familyId, activations, maxActivation) {
    container.innerHTML = '<div class="viewer-placeholder">Scroll to load 3D structure</div>';
    container.dataset.loaded = "false";

    const observer = new IntersectionObserver(
        (entries) => {
            for (const entry of entries) {
                if (entry.isIntersecting && container.dataset.loaded === "false") {
                    container.dataset.loaded = "true";
                    container.innerHTML = '<div class="viewer-placeholder"><div class="loading-spinner"></div> Loading structure...</div>';
                    createNmpfamMolViewer(container, familyId, activations, maxActivation);
                    observer.unobserve(container);
                }
            }
        },
        { rootMargin: "200px" }
    );
    observer.observe(container);
}

/**
 * Create a 3Dmol viewer for an NMPFams protein, colored by SAE activation.
 */
async function createNmpfamMolViewer(container, familyId, activations, maxActivation) {
    try {
        const res = await fetch(`/api/nmpfam-pdb/${familyId}`);
        if (!res.ok) {
            container.innerHTML = '<div class="viewer-placeholder">No structure available</div>';
            return;
        }
        const pdbData = await res.text();

        container.innerHTML = "";
        const viewer = $3Dmol.createViewer(container, {
            backgroundColor: "white",
            antialias: true,
        });
        viewer.addModel(pdbData, "pdb");

        // Color by activation (blue=low, red=high)
        const atoms = viewer.getModel().selectedAtoms({});
        if (activations && activations.length > 0) {
            const colorMap = {};
            for (let i = 0; i < activations.length; i++) {
                const norm = maxActivation > 0 ? activations[i] / maxActivation : 0;
                const r = Math.round(255 * Math.min(1, norm * 2));
                const b = Math.round(255 * Math.max(0, 1 - norm * 2));
                colorMap[i + 1] = `rgb(${r},0,${b})`;
            }
            viewer.setStyle({}, {
                cartoon: {
                    colorfunc: function(atom) {
                        return colorMap[atom.resi] || "rgb(200,200,200)";
                    }
                }
            });
        } else {
            viewer.setStyle({}, { cartoon: { color: "spectrum" } });
        }

        viewer.zoomTo();
        viewer.render();
    } catch (e) {
        container.innerHTML = '<div class="viewer-placeholder">Failed to load structure</div>';
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
        // Fetch all endpoints in parallel
        // InterPro, Geometry, Motif, Position may 404 — that's expected for missing enrichment
        const [featureRes, interproRes, geometryRes, motifRes, positionRes, cathRes, gpRes, nmpfamRes] = await Promise.all([
            fetch(`/api/feature/${featureId}`),
            fetch(`/api/feature/${featureId}/interpro`).catch(() => null),
            fetch(`/api/feature/${featureId}/geometry`).catch(() => null),
            fetch(`/api/feature/${featureId}/motif`).catch(() => null),
            fetch(`/api/feature/${featureId}/position`).catch(() => null),
            fetch(`/api/feature/${featureId}/cath`).catch(() => null),
            fetch(`/api/geometry-primary`).catch(() => null),
            fetch(`/api/feature/${featureId}/nmpfam`).catch(() => null),
        ]);

        if (!featureRes.ok) {
            throw new Error(`Feature fetch failed: ${featureRes.status}`);
        }

        const featureData = await featureRes.json();
        const interproData = interproRes && interproRes.ok ? await interproRes.json() : null;
        const geometryData = geometryRes && geometryRes.ok ? await geometryRes.json() : null;
        const motifData = motifRes && motifRes.ok ? await motifRes.json() : null;
        const positionData = positionRes && positionRes.ok ? await positionRes.json() : null;
        const cathData = cathRes && cathRes.ok ? await cathRes.json() : null;
        const nmpfamData = nmpfamRes && nmpfamRes.ok ? await nmpfamRes.json() : null;

        // Extract this feature's geometry-primary info
        let gpInfo = null;
        if (gpRes && gpRes.ok) {
            const gpData = await gpRes.json();
            gpInfo = (gpData.features || {})[String(featureId)] || null;
        }

        const featureMaxAct = featureData.max_activation || 1;
        const bestAnnotationName = getBestAnnotationName(interproData);

        // Section 1: Summary stats
        renderSummaryCards(
            document.getElementById("summary-cards"),
            featureData, interproData, geometryData, motifData, positionData, cathData, gpInfo
        );

        // Geometry Radar Profile (between summary and sequences)
        if (geometryData) {
            const geo = geometryData.geometric_residue_level;
            const importances = geo && geo.feature_importances;
            if (importances && typeof aggregateToCategories === "function") {
                const scores = aggregateToCategories(importances);
                if (scores) {
                    document.getElementById("radar-section").style.display = "";
                    renderRadarWithLegend(
                        document.getElementById("radar-container"),
                        scores,
                        { size: 240 }
                    );
                }
            }
        }

        // Section 1b: Top 3 sequence alignment
        renderAlignment(
            document.getElementById("alignment-container"),
            featureData
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

        // Section 4: NMPFams novel protein hits
        if (nmpfamData && nmpfamData.nmpfam_hits && nmpfamData.nmpfam_hits.length > 0) {
            document.getElementById("nmpfam-section").style.display = "";
            renderNmpfamHits(
                document.getElementById("nmpfam-container"),
                nmpfamData, featureMaxAct
            );
        }

        // Section 5: Geometry plots (only if data exists)
        if (geometryData) {
            document.getElementById("geometry-section").style.display = "";
            renderGeometryPlots(
                document.getElementById("geometry-container"),
                geometryData,
                nmpfamData
            );
        }
    } catch (err) {
        console.error("Failed to load feature data:", err);
        document.getElementById("summary-cards").innerHTML =
            `<p style="color:red">Error loading feature: ${err.message}</p>`;
    }
});
