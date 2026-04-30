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
// Method-annotation grid (Section 1)
// ============================================================

/** Seven methods in paper order; keep this in sync with homepage.js METHOD_DEFS. */
const METHOD_DEFS = [
    { id: 1, name: "InterPro Protein",    metric: "F1",     scrollTo: "#top-proteins-section" },
    { id: 2, name: "InterPro Residue",    metric: "F1",     scrollTo: "#alignment-section"     },
    { id: 3, name: "CATH Protein",        metric: "F1",     scrollTo: "#top-proteins-section"  },
    { id: 4, name: "CATH Residue",        metric: "F1",     scrollTo: "#alignment-section"     },
    { id: 5, name: "Sequence Position",   metric: "F1",     scrollTo: "#alignment-section"     },
    { id: 6, name: "Sequence MEME Motif", metric: "PR-AUC", scrollTo: "#alignment-section"     },
    { id: 7, name: "Geometric",           metric: "PR-AUC", scrollTo: "#geometry-section"      },
];

const isSig = (q) => q != null && q < 0.05;

function fmtQ(q) {
    if (q == null) return "—";
    if (q < 1e-3) return q.toExponential(1);
    return q.toFixed(3);
}

/**
 * Render the 7-card method grid for a feature.
 *
 * `sig` is the object returned by /api/feature/{id}/significance:
 * { feature_id, m1_score, m1_label, m1_q, ..., m7_score, m7_label, m7_q }.
 */
function renderMethodGrid(container, sig) {
    container.innerHTML = "";
    for (const def of METHOD_DEFS) {
        const k = def.id;
        const score = sig[`m${k}_score`];
        const label = sig[`m${k}_label`];
        const q = sig[`m${k}_q`];
        const significant = isSig(q);

        const card = document.createElement("article");
        card.className = "method-card" + (significant ? " is-significant" : " not-significant");
        card.innerHTML = `
            <div class="mc-name">${def.name}</div>
            <div class="mc-metric">${def.metric}</div>
            <div class="mc-score">${fmtVal(score, 3)}</div>
            <div class="mc-label">${label || "—"}</div>
            <span class="mc-q">q = ${fmtQ(q)}</span>
        `;
        card.addEventListener("click", () => {
            const el = document.querySelector(def.scrollTo);
            if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
        });
        container.appendChild(card);
    }
}

/** Utility: format a number or return "—" */
function fmtVal(v, decimals = 3) {
    if (v === null || v === undefined) return "—";
    return Number(v).toFixed(decimals);
}

// ============================================================
// Section 2: Top activating proteins
// ============================================================

/**
 * Render the top 5 most activating protein entries.
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
    const threshold = nmpfamData.activation_threshold_sae ?? nmpfamData.activation_threshold ?? 0;

    // Summary line
    const summary = document.createElement("p");
    summary.innerHTML = `<strong>${hits.length}</strong> novel metagenomic famil${hits.length === 1 ? "y" : "ies"} activated above threshold (${fmtVal(threshold, 3)})`;
    container.appendChild(summary);

    for (const hit of hits) {
        const entry = document.createElement("div");
        entry.className = "protein-entry";

        const maxAct = hit.max_sae_activation ?? hit.max_activation;
        const nRes = hit.n_residues ?? hit.sequence_length ?? hit.sequence?.length ?? "?";
        const acts = hit.sae_activation_profile ?? hit.per_residue_activations;

        // Label with family info and link
        const label = document.createElement("div");
        label.className = "protein-label";
        label.innerHTML = `<a href="${hit.nmpfams_url}" target="_blank">${hit.family_id}</a> · `
            + `max: ${fmtVal(maxAct, 4)} · `
            + `${nRes} residues · `
            + `<span style="opacity:0.7">${hit.category} · ${hit.sequence_count} members</span>`;
        entry.appendChild(label);

        // Sequence strip with activation coloring
        if (acts && hit.sequence) {
            // Text sequence
            createTextSequence(entry, {
                sequence: hit.sequence,
                activations: acts,
                maxActivation: featureMaxAct,
                accession: hit.family_id,
                maxAct: maxAct,
                showLabel: false,
            });

            // Canvas strip
            const stripDiv = document.createElement("div");
            entry.appendChild(stripDiv);
            createSequenceStrip(stripDiv, {
                sequence: hit.sequence,
                activations: acts,
                maxActivation: featureMaxAct,
                accession: hit.family_id,
            });
        }

        // 3D viewer from NMPFams PDB proxy (lazy; shows "No structure" if 404)
        const viewerDiv = document.createElement("div");
        viewerDiv.className = "viewer-container";
        entry.appendChild(viewerDiv);
        lazyLoadNmpfamViewer(
            viewerDiv, hit.family_id,
            acts || [],
            featureMaxAct
        );

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
        // Fetch all endpoints in parallel. Enrichment endpoints may 404 —
        // that's expected for features missing a specific analysis.
        const [featureRes, sigRes, interproRes, geometryRes, motifRes, positionRes, cathRes, nmpfamRes] = await Promise.all([
            fetch(`/api/feature/${featureId}`),
            fetch(`/api/feature/${featureId}/significance`),
            fetch(`/api/feature/${featureId}/interpro`).catch(() => null),
            fetch(`/api/feature/${featureId}/geometry`).catch(() => null),
            fetch(`/api/feature/${featureId}/motif`).catch(() => null),
            fetch(`/api/feature/${featureId}/position`).catch(() => null),
            fetch(`/api/feature/${featureId}/cath`).catch(() => null),
            fetch(`/api/feature/${featureId}/nmpfam`).catch(() => null),
        ]);

        if (!featureRes.ok) {
            throw new Error(`Feature fetch failed: ${featureRes.status}`);
        }

        const featureData = await featureRes.json();
        const sig = sigRes.ok ? await sigRes.json() : null;
        const interproData = interproRes && interproRes.ok ? await interproRes.json() : null;
        const geometryData = geometryRes && geometryRes.ok ? await geometryRes.json() : null;
        const motifData = motifRes && motifRes.ok ? await motifRes.json() : null;
        const positionData = positionRes && positionRes.ok ? await positionRes.json() : null;
        const cathData = cathRes && cathRes.ok ? await cathRes.json() : null;
        const nmpfamData = nmpfamRes && nmpfamRes.ok ? await nmpfamRes.json() : null;

        const featureMaxAct = featureData.max_activation || 1;
        const bestAnnotationName = getBestAnnotationName(interproData);

        // Section 1: 7-card method grid
        const methodGrid = document.getElementById("method-grid");
        if (sig) {
            renderMethodGrid(methodGrid, sig);
        } else {
            methodGrid.innerHTML = '<p class="secondary">Significance data not available for this feature.</p>';
        }

        // Geometric radar sits beside the method grid.
        // renderRadarWithLegend expects a 6-element array ordered
        // [curvature, torsion, planarity, compactness, contacts, composition].
        const radarSlot = document.getElementById("radar-slot");
        const radarContainer = document.getElementById("radar-container");
        if (sig && sig.geometry_radar) {
            const r = sig.geometry_radar;
            const scores = [
                r.curvature || 0, r.torsion || 0, r.planarity || 0,
                r.compactness || 0, r.contacts || 0, r.composition || 0,
            ];
            radarSlot.style.display = "";
            renderRadarWithLegend(radarContainer, scores, { size: 180 });
        } else if (geometryData) {
            const geo = geometryData.geometric_residue_level;
            const importances = geo && geo.feature_importances;
            if (importances && typeof aggregateToCategories === "function") {
                const scores = aggregateToCategories(importances);
                if (scores) {
                    radarSlot.style.display = "";
                    renderRadarWithLegend(radarContainer, scores, { size: 180 });
                }
            }
        }

        // Top activating sequences + bins + NMPFams + geometry plots unchanged.
        renderAlignment(document.getElementById("alignment-container"), featureData);
        renderTopProteins(
            document.getElementById("top-proteins-container"),
            featureData, interproData
        );
        renderBins(
            document.getElementById("bins-container"),
            featureData, featureMaxAct, bestAnnotationName
        );
        if (nmpfamData && nmpfamData.nmpfam_hits && nmpfamData.nmpfam_hits.length > 0) {
            document.getElementById("nmpfam-section").style.display = "";
            renderNmpfamHits(
                document.getElementById("nmpfam-container"),
                nmpfamData, featureMaxAct
            );
        }
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
        const mg = document.getElementById("method-grid");
        if (mg) mg.innerHTML = `<p style="color:red">Error loading feature: ${err.message}</p>`;
    }
});
