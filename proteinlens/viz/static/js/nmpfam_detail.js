/**
 * nmpfam_detail.js — Detail page for a single triple-intersection feature.
 *
 * Shows the feature's NMPFams hits with activation+geometry plots, 3D viewers,
 * and comparison to the SwissProt top-activating proteins.
 */

function fmtVal(v, d = 3) {
    if (v == null || v === undefined) return "\u2014";
    return typeof v === "number" ? v.toFixed(d) : String(v);
}

function getFeatureIdFromUrl() {
    const parts = window.location.pathname.split("/");
    const idStr = parts[parts.length - 1];
    const id = parseInt(idStr, 10);
    return isNaN(id) ? null : id;
}

function createStatCard(title, inner) {
    const card = document.createElement("article");
    card.className = "stat-card";
    card.innerHTML = `<header><strong>${title}</strong></header><div>${inner}</div>`;
    return card;
}

// ── Summary cards ──

function renderSummaryCards(container, caseFeature, featureData, geometryData) {
    container.innerHTML = "";

    const geom = caseFeature.geometry || {};
    const ipro = caseFeature.interpro || {};
    const conc = geom.concordance || {};

    container.appendChild(createStatCard("Feature Identity",
        `<div class="value"><a href="/feature/${caseFeature.feature_id}">Feature ${caseFeature.feature_id}</a></div>
         <div class="detail">Coverage: ${fmtVal(caseFeature.coverage_pct, 1)}% (sparse)</div>
         <div class="detail">Composite score: ${fmtVal(caseFeature.composite_score, 3)}</div>
         <div class="detail">Global max activation: ${fmtVal(caseFeature.global_max_activation, 3)}</div>`
    ));

    container.appendChild(createStatCard("NMPFams Hits",
        `<div class="value">${caseFeature.n_nmpfam_hits} families</div>
         <div class="detail">Top activation: ${fmtVal(caseFeature.top_nmpfam_norm_act * 100, 1)}% of global max</div>
         <div class="detail">Threshold: ${fmtVal(caseFeature.activation_threshold, 3)}</div>`
    ));

    container.appendChild(createStatCard("Geometry Classifier",
        `<div class="value">AUC ${fmtVal(geom.gbm_auc_cv, 3)}</div>
         <div class="detail">F1-CV: ${fmtVal(geom.tree_f1_cv, 3)}</div>
         <div class="detail">Concordance Spearman: ${fmtVal(conc.spearman_r, 3)}</div>
         <div class="detail">Concordance F1: ${fmtVal(conc.concordance_f1, 3)}</div>`
    ));

    container.appendChild(createStatCard("InterPro",
        `<div class="detail">Protein F1: ${fmtVal(ipro.protein_best_f1, 3)}</div>
         <div class="detail">${ipro.protein_best_name || "No annotation"}</div>
         <div class="detail">Residue F1: ${fmtVal(ipro.residue_best_f1, 3)}</div>`
    ));
}

// ── NMPFams hit rendering (with geometry plots) ──

function renderNmpfamHits(container, nmpfamData, featureMaxAct) {
    container.innerHTML = "";

    const hits = nmpfamData.nmpfam_hits || [];
    if (hits.length === 0) {
        container.innerHTML = '<p class="secondary">No NMPFams hits.</p>';
        return;
    }

    for (const hit of hits) {
        const entry = document.createElement("div");
        entry.className = "protein-entry";

        // Label
        const label = document.createElement("div");
        label.className = "protein-label";
        label.innerHTML = `<a href="${hit.nmpfams_url}" target="_blank">${hit.family_id}</a> &middot; `
            + `max: ${fmtVal(hit.max_activation, 4)} (${fmtVal(hit.normalized_activation * 100, 1)}% of global max) &middot; `
            + `${hit.sequence_length || "?"} residues &middot; `
            + `<span style="opacity:0.7">${hit.category} &middot; ${hit.sequence_count} members</span>`;
        entry.appendChild(label);

        // Dual-axis activation vs geometry plot + concordance strip
        if (hit.sae_activation_profile && hit.geom_prob_profile) {
            const overlayDiv = document.createElement("div");
            overlayDiv.className = "plot-container";
            entry.appendChild(overlayDiv);
            renderDualAxisPlot(overlayDiv, hit);

            if (hit.concordance_labels && hit.concordance_labels.length > 0) {
                const concordDiv = document.createElement("div");
                concordDiv.style.marginBottom = "0.75rem";
                entry.appendChild(concordDiv);
                renderConcordanceStrip(concordDiv, hit.concordance_labels);
            }
        }

        // Sequence strip
        if (hit.per_residue_activations && hit.sequence) {
            createTextSequence(entry, {
                sequence: hit.sequence,
                activations: hit.per_residue_activations,
                maxActivation: featureMaxAct,
                accession: hit.family_id,
                maxAct: hit.max_activation,
                showLabel: false,
            });

            const stripDiv = document.createElement("div");
            entry.appendChild(stripDiv);
            createSequenceStrip(stripDiv, {
                sequence: hit.sequence,
                activations: hit.per_residue_activations,
                maxActivation: featureMaxAct,
                accession: hit.family_id,
            });
        }

        // 3D viewer
        if (hit.pdb_available) {
            const viewerDiv = document.createElement("div");
            viewerDiv.className = "viewer-container";
            entry.appendChild(viewerDiv);
            lazyLoadNmpfamDetailViewer(viewerDiv, hit.family_id, hit.per_residue_activations || [], featureMaxAct);
        }

        container.appendChild(entry);
    }
}

function lazyLoadNmpfamDetailViewer(container, familyId, activations, maxAct) {
    container.innerHTML = '<div class="viewer-placeholder">Scroll to load 3D structure</div>';
    container.dataset.loaded = "false";
    const obs = new IntersectionObserver((entries) => {
        for (const e of entries) {
            if (e.isIntersecting && container.dataset.loaded === "false") {
                container.dataset.loaded = "true";
                container.innerHTML = '<div class="viewer-placeholder"><div class="loading-spinner"></div> Loading...</div>';
                loadNmpfamViewer(container, familyId, activations, maxAct);
                obs.unobserve(container);
            }
        }
    }, { rootMargin: "200px" });
    obs.observe(container);
}

async function loadNmpfamViewer(container, familyId, activations, maxAct) {
    try {
        const res = await fetch(`/api/nmpfam-pdb/${familyId}`);
        if (!res.ok) { container.innerHTML = '<div class="viewer-placeholder">No structure</div>'; return; }
        const pdb = await res.text();
        container.innerHTML = "";
        const viewer = $3Dmol.createViewer(container, { backgroundColor: "white", antialias: true });
        viewer.addModel(pdb, "pdb");
        if (activations && activations.length > 0) {
            const colorMap = {};
            for (let i = 0; i < activations.length; i++) {
                const n = maxAct > 0 ? activations[i] / maxAct : 0;
                colorMap[i + 1] = `rgb(${Math.round(255 * Math.min(1, n * 2))},0,${Math.round(255 * Math.max(0, 1 - n * 2))})`;
            }
            viewer.setStyle({}, { cartoon: { colorfunc: (atom) => colorMap[atom.resi] || "rgb(200,200,200)" } });
        } else {
            viewer.setStyle({}, { cartoon: { color: "spectrum" } });
        }
        viewer.zoomTo(); viewer.render();
    } catch (e) {
        container.innerHTML = '<div class="viewer-placeholder">Failed to load</div>';
    }
}

// ── SwissProt comparison ──

function renderSwissprotProteins(container, featureData) {
    container.innerHTML = "";
    const topSeqs = (featureData.top_sequences || []).slice(0, 3);
    if (topSeqs.length === 0) {
        container.innerHTML = '<p class="secondary">No SwissProt activating proteins.</p>';
        return;
    }

    const featureMaxAct = featureData.max_activation || 1;
    for (const protein of topSeqs) {
        const entry = document.createElement("div");
        entry.className = "protein-entry";

        const label = document.createElement("div");
        label.className = "protein-label";
        label.textContent = `${protein.accession} \u00b7 max: ${fmtVal(protein.max_activation, 4)} \u00b7 ${protein.sequence_length || protein.sequence?.length || "?"} residues`;
        entry.appendChild(label);

        if (protein.per_residue_activations && protein.sequence) {
            createTextSequence(entry, {
                sequence: protein.sequence,
                activations: protein.per_residue_activations,
                maxActivation: featureMaxAct,
                accession: protein.accession,
                maxAct: protein.max_activation,
                showLabel: false,
            });

            const stripDiv = document.createElement("div");
            entry.appendChild(stripDiv);
            createSequenceStrip(stripDiv, {
                sequence: protein.sequence,
                activations: protein.per_residue_activations,
                maxActivation: featureMaxAct,
                accession: protein.accession,
            });
        }

        if (protein.pdb_available !== false) {
            const viewerDiv = document.createElement("div");
            viewerDiv.className = "viewer-container";
            entry.appendChild(viewerDiv);
            lazyLoadViewer(viewerDiv, protein.accession, protein.per_residue_activations || [], featureMaxAct);
        }

        container.appendChild(entry);
    }
}

// ── Main ──

document.addEventListener("DOMContentLoaded", async () => {
    const featureId = getFeatureIdFromUrl();
    if (featureId === null) {
        document.getElementById("page-title").textContent = "Error: Invalid feature ID";
        return;
    }

    document.getElementById("page-title").textContent = `NMPFams \u2014 Feature ${featureId}`;
    document.title = `NMPFams Feature ${featureId} \u2014 SAE Visualizer`;

    try {
        const [caseRes, featureRes, nmpfamRes, geometryRes] = await Promise.all([
            fetch("/api/nmpfam-case-study"),
            fetch(`/api/feature/${featureId}`),
            fetch(`/api/feature/${featureId}/nmpfam`).catch(() => null),
            fetch(`/api/feature/${featureId}/geometry`).catch(() => null),
        ]);

        if (!caseRes.ok) throw new Error("Case study data not found");
        if (!featureRes.ok) throw new Error(`Feature ${featureId} not found`);

        const caseData = await caseRes.json();
        const featureData = await featureRes.json();
        const nmpfamData = nmpfamRes && nmpfamRes.ok ? await nmpfamRes.json() : null;
        const geometryData = geometryRes && geometryRes.ok ? await geometryRes.json() : null;

        // Find this feature in the case study
        const caseFeature = (caseData.triple_features || []).find(f => f.feature_id === featureId);
        if (!caseFeature) {
            document.getElementById("summary-cards").innerHTML =
                `<p>Feature ${featureId} is not in the triple intersection. <a href="/feature/${featureId}">View feature page</a></p>`;
            return;
        }

        const featureMaxAct = featureData.max_activation || 1;

        // Summary cards
        renderSummaryCards(document.getElementById("summary-cards"), caseFeature, featureData, geometryData);

        // Radar
        if (geometryData) {
            const imp = geometryData.geometric_residue_level?.feature_importances;
            if (imp && typeof aggregateToCategories === "function") {
                const scores = aggregateToCategories(imp);
                if (scores) {
                    document.getElementById("radar-section").style.display = "";
                    renderRadarWithLegend(document.getElementById("radar-container"), scores, { size: 240 });
                }
            }
        }

        // NMPFams hits
        if (nmpfamData) {
            renderNmpfamHits(document.getElementById("nmpfam-hits-container"), nmpfamData, featureMaxAct);
        } else {
            document.getElementById("nmpfam-hits-container").innerHTML = '<p class="secondary">NMPFams enrichment data not available.</p>';
        }

        // SwissProt comparison
        renderSwissprotProteins(document.getElementById("swissprot-container"), featureData);

        // Geometry plots
        if (geometryData) {
            document.getElementById("geometry-section").style.display = "";
            renderGeometryPlots(document.getElementById("geometry-container"), geometryData, nmpfamData);
        }

    } catch (err) {
        console.error(err);
        document.getElementById("summary-cards").innerHTML = `<p style="color:red">Error: ${err.message}</p>`;
    }
});
