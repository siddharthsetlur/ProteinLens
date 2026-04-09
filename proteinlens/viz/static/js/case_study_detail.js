/**
 * case_study_detail.js — Renders the case study detail page for a single family.
 *
 * URL: /case-studies/{annotation_code}
 *
 * Shows:
 *   1. Geometry feature importance heatmap across nodes
 *   2. Per-node top protein with activation profile + InterPro overlay
 *   3. Cross-node activation overlay on shared proteins
 */

function fmtVal(v, decimals = 3) {
    if (v === null || v === undefined) return "\u2014";
    return Number(v).toFixed(decimals);
}

function getAnnotationCodeFromUrl() {
    const parts = window.location.pathname.split("/");
    return decodeURIComponent(parts[parts.length - 1]);
}

// ============================================================
// 1. Geometry Feature Importance Heatmap
// ============================================================

function renderHeatmap(container, family, geomFeatureNames) {
    // Build importance matrix: rows = nodes, columns = geometry features
    // Only include features with non-negligible importance in at least one node
    const members = family.members;

    // Filter to features with max importance > 0.02
    const featureSums = {};
    for (const fn of geomFeatureNames) {
        featureSums[fn] = Math.max(...members.map(m => (m.feature_importances || {})[fn] || 0));
    }
    const activeFeatures = geomFeatureNames.filter(fn => featureSums[fn] > 0.02);

    // Sort features by total importance (descending)
    activeFeatures.sort((a, b) => {
        const sumA = members.reduce((s, m) => s + ((m.feature_importances || {})[a] || 0), 0);
        const sumB = members.reduce((s, m) => s + ((m.feature_importances || {})[b] || 0), 0);
        return sumB - sumA;
    });

    const zValues = members.map(m =>
        activeFeatures.map(fn => (m.feature_importances || {})[fn] || 0)
    );

    const yLabels = members.map(m =>
        `Node ${m.feature_id} (${m.top_geometric_feature})`
    );

    const trace = {
        z: zValues,
        x: activeFeatures,
        y: yLabels,
        type: "heatmap",
        colorscale: [
            [0, "#ffffff"],
            [0.2, "#fee8c8"],
            [0.4, "#fdbb84"],
            [0.6, "#e34a33"],
            [1.0, "#7a0177"],
        ],
        colorbar: { title: "Importance", titleside: "right" },
        hovertemplate: "%{y}<br>%{x}: %{z:.4f}<extra></extra>",
    };

    const layout = {
        height: Math.max(200, members.length * 60 + 120),
        margin: { t: 20, b: 120, l: 280, r: 80 },
        xaxis: {
            tickangle: -45,
            tickfont: { size: 10 },
        },
        yaxis: {
            tickfont: { size: 11 },
            autorange: "reversed",
        },
    };

    Plotly.newPlot(container, [trace], layout, { responsive: true, displayModeBar: false });
}

// ============================================================
// 2. Per-Node Comparison
// ============================================================

function renderNodeComparison(container, member, featureData, interproData, geometryData) {
    const section = document.createElement("div");
    section.style.marginBottom = "2rem";
    section.style.padding = "1rem";
    section.style.border = "1px solid var(--pico-muted-border-color)";
    section.style.borderRadius = "8px";

    // Header with node ID, metrics, and radar glyph
    const header = document.createElement("div");
    header.style.marginBottom = "0.75rem";
    header.innerHTML = `
        <div style="display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;">
            <a href="/feature/${member.feature_id}" style="font-size:1.2rem;font-weight:700;">Node ${member.feature_id}</a>
            <span id="radar-node-${member.feature_id}" style="flex-shrink:0;"></span>
            <span class="badge badge-done">InterPro F1: ${fmtVal(member.interpro_res_f1)}</span>
            <span class="badge badge-done">Geom PR-AUC: ${fmtVal(member.geom_pr_auc)}</span>
            <span class="badge badge-count" style="font-family:monospace">${member.top_geometric_feature}</span>
            <span class="badge badge-count">${member.structural_category}</span>
        </div>
    `;
    section.appendChild(header);

    // Render inline radar glyph from member's feature importances
    if (member.feature_importances && typeof aggregateToCategories === "function") {
        const scores = aggregateToCategories(member.feature_importances);
        if (scores) {
            const radarSpan = header.querySelector(`#radar-node-${member.feature_id}`);
            renderRadarGlyph(radarSpan, scores, { size: 52, showLabels: false });
        }
    }

    // Decision tree rules (compact)
    if (member.rules) {
        const rulesDiv = document.createElement("details");
        rulesDiv.style.marginBottom = "0.75rem";
        rulesDiv.style.fontSize = "0.8rem";
        rulesDiv.innerHTML = `
            <summary style="cursor:pointer;font-weight:600;">Decision Tree Rules</summary>
            <pre style="background:#f5f5f5;padding:0.5rem;border-radius:4px;overflow-x:auto;font-size:0.75rem;margin:0.5rem 0 0 0;">${member.rules}</pre>
        `;
        section.appendChild(rulesDiv);
    }

    // Top 3 protein entries
    const topSeqs = (featureData.top_sequences || []).slice(0, 3);
    const featureMaxAct = featureData.max_activation || 1;

    // Get best residue-level annotation name for domain overlay
    let bestAnnotationName = null;
    if (interproData) {
        const resEntries = interproData.residue_level || [];
        if (resEntries.length > 0) {
            bestAnnotationName = resEntries.reduce((a, b) =>
                ((b.best_f1 || 0) > (a.best_f1 || 0) ? b : a), resEntries[0]
            ).annotation_name || null;
        }
    }

    for (const protein of topSeqs) {
        const entry = document.createElement("div");
        entry.className = "protein-entry";

        // Label
        const label = document.createElement("div");
        label.className = "protein-label";
        label.textContent = `${protein.accession} \u00b7 max: ${fmtVal(protein.max_activation, 4)} \u00b7 ${protein.sequence?.length || "?"} residues`;
        entry.appendChild(label);

        // Text sequence with activation coloring
        createTextSequence(entry, {
            sequence: protein.sequence || "",
            activations: protein.per_residue_activations || [],
            maxActivation: featureMaxAct,
            accession: protein.accession,
            maxAct: protein.max_activation,
            showLabel: false,
        });

        // Sequence strip with InterPro overlay
        const stripDiv = document.createElement("div");
        entry.appendChild(stripDiv);
        createSequenceStrip(stripDiv, {
            sequence: protein.sequence || "",
            activations: protein.per_residue_activations || [],
            maxActivation: featureMaxAct,
            accession: protein.accession,
            bestAnnotationName: bestAnnotationName,
        });

        section.appendChild(entry);
    }

    // Geometry activation vs geom prob overlay for top protein
    if (geometryData && geometryData.plot_data && geometryData.plot_data.top_proteins) {
        const geoProtein = geometryData.plot_data.top_proteins[0];
        if (geoProtein && geoProtein.sae_activation_profile) {
            const plotDiv = document.createElement("div");
            plotDiv.className = "plot-container";
            section.appendChild(plotDiv);
            renderDualAxisPlot(plotDiv, geoProtein);
        }
    }

    container.appendChild(section);
}

// ============================================================
// 3. Cross-Node Activation Overlay
// ============================================================

function renderCrossNodeOverlay(container, memberData) {
    // Find proteins that appear in multiple nodes' top sequences
    const proteinNodes = {}; // accession -> [{feature_id, activations, sequence}]
    for (const { member, featureData } of memberData) {
        for (const protein of (featureData.top_sequences || [])) {
            const key = protein.accession;
            if (!proteinNodes[key]) proteinNodes[key] = [];
            proteinNodes[key].push({
                feature_id: member.feature_id,
                activations: protein.per_residue_activations || [],
                sequence: protein.sequence || "",
                max_activation: protein.max_activation,
                top_geom: member.top_geometric_feature,
            });
        }
    }

    // Find shared proteins (appear in 2+ nodes)
    const shared = Object.entries(proteinNodes)
        .filter(([, nodes]) => nodes.length >= 2)
        .sort((a, b) => b[1].length - a[1].length);

    if (shared.length === 0) {
        // Fall back: overlay top protein from each node
        renderFallbackOverlay(container, memberData);
        return;
    }

    // Plot up to 3 shared proteins
    for (const [accession, nodes] of shared.slice(0, 3)) {
        const plotDiv = document.createElement("div");
        plotDiv.style.marginBottom = "1.5rem";
        container.appendChild(plotDiv);

        const seqLen = nodes[0].activations.length;
        const xPositions = Array.from({ length: seqLen }, (_, i) => i + 1);

        const palette = ["#dc2626", "#2563eb", "#16a34a", "#d97706", "#7c3aed", "#db2777"];
        const traces = nodes.map((n, i) => ({
            x: xPositions,
            y: n.activations,
            name: `Node ${n.feature_id} (${n.top_geom})`,
            type: "scatter",
            mode: "lines",
            line: { color: palette[i % palette.length], width: 1.5 },
        }));

        const layout = {
            title: { text: `${accession} — ${nodes.length} nodes`, font: { size: 13 } },
            xaxis: { title: "Residue Position" },
            yaxis: { title: "SAE Activation" },
            height: 300,
            margin: { t: 40, b: 40, l: 60, r: 20 },
            legend: { x: 0.01, y: 0.99, bgcolor: "rgba(255,255,255,0.8)" },
        };

        Plotly.newPlot(plotDiv, traces, layout, { responsive: true, displayModeBar: false });
    }
}

function renderFallbackOverlay(container, memberData) {
    // No shared proteins — show top protein from each node overlaid
    // Use the first node's top protein as reference
    const plotDiv = document.createElement("div");
    container.appendChild(plotDiv);

    const palette = ["#dc2626", "#2563eb", "#16a34a", "#d97706", "#7c3aed", "#db2777"];
    const traces = [];

    for (let i = 0; i < memberData.length; i++) {
        const { member, featureData } = memberData[i];
        const topProtein = (featureData.top_sequences || [])[0];
        if (!topProtein) continue;
        const acts = topProtein.per_residue_activations || [];
        const xPositions = Array.from({ length: acts.length }, (_, j) => j + 1);
        traces.push({
            x: xPositions,
            y: acts,
            name: `Node ${member.feature_id}: ${topProtein.accession} (${member.top_geometric_feature})`,
            type: "scatter",
            mode: "lines",
            line: { color: palette[i % palette.length], width: 1.5 },
        });
    }

    const layout = {
        title: { text: "Top Protein Activation Profiles (different proteins)", font: { size: 13 } },
        xaxis: { title: "Residue Position" },
        yaxis: { title: "SAE Activation" },
        height: 350,
        margin: { t: 40, b: 40, l: 60, r: 20 },
        legend: { x: 0.01, y: 0.99, bgcolor: "rgba(255,255,255,0.8)" },
    };

    Plotly.newPlot(plotDiv, traces, layout, { responsive: true, displayModeBar: false });
}

// ============================================================
// Main: fetch data and render
// ============================================================

document.addEventListener("DOMContentLoaded", async () => {
    const annotationCode = getAnnotationCodeFromUrl();
    if (!annotationCode) {
        document.getElementById("page-title").textContent = "Error: No annotation code";
        return;
    }

    try {
        // Fetch family data
        const familiesRes = await fetch("/api/case-study-families");
        if (!familiesRes.ok) throw new Error(`Failed to fetch families: ${familiesRes.status}`);
        const familiesData = await familiesRes.json();

        // Find the specific family
        const family = familiesData.families.find(f => f.annotation_code === annotationCode);
        if (!family) throw new Error(`Family ${annotationCode} not found`);

        document.getElementById("page-title").textContent = family.annotation_name;
        document.title = `${family.annotation_name} \u2014 Case Study`;

        // Family summary
        document.getElementById("family-info").innerHTML = `
            <strong>${family.annotation_code}</strong> &mdash;
            ${family.n_nodes} SAE nodes share this residue-level InterPro annotation.
            Mean geometry cosine similarity: <strong>${fmtVal(family.mean_cosine_similarity, 3)}</strong>
            (lower = more diverse geometry).
            ${family.geom_diverse ? "Nodes use <strong>different</strong> top geometric features." : "Nodes share the <strong>same</strong> top geometric feature."}
        `;

        // 1. Heatmap
        renderHeatmap(
            document.getElementById("heatmap-container"),
            family,
            familiesData.geometry_feature_names || []
        );

        // 2 & 3. Fetch per-node data in parallel
        const fetchPromises = family.members.map(async (member) => {
            const [featureRes, interproRes, geometryRes] = await Promise.all([
                fetch(`/api/feature/${member.feature_id}`),
                fetch(`/api/feature/${member.feature_id}/interpro`).catch(() => null),
                fetch(`/api/feature/${member.feature_id}/geometry`).catch(() => null),
            ]);
            return {
                member,
                featureData: featureRes.ok ? await featureRes.json() : { top_sequences: [] },
                interproData: interproRes && interproRes.ok ? await interproRes.json() : null,
                geometryData: geometryRes && geometryRes.ok ? await geometryRes.json() : null,
            };
        });

        const memberData = await Promise.all(fetchPromises);

        // Render per-node comparison
        const nodesContainer = document.getElementById("nodes-container");
        nodesContainer.innerHTML = "";
        for (const { member, featureData, interproData, geometryData } of memberData) {
            renderNodeComparison(nodesContainer, member, featureData, interproData, geometryData);
        }

        // Render cross-node overlay
        renderCrossNodeOverlay(
            document.getElementById("overlay-container"),
            memberData
        );

    } catch (err) {
        console.error("Failed to load case study:", err);
        document.getElementById("nodes-container").innerHTML =
            `<p style="color:red">Error: ${err.message}</p>`;
    }
});
