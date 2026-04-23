/**
 * meme_case_study_detail.js — Renders the MEME case study detail page.
 *
 * URL: /meme-case-studies/{representative_consensus}
 *
 * Shows:
 *   1. Best MEME consensus per node
 *   2. Geometry feature importance heatmap across nodes
 *   3. Per-node top proteins with activation profiles
 *   4. Cross-node activation overlay on shared proteins
 */

function fmtVal(v, decimals = 3) {
    if (v === null || v === undefined) return "\u2014";
    return Number(v).toFixed(decimals);
}

function getConsensusFromUrl() {
    const parts = window.location.pathname.split("/");
    return decodeURIComponent(parts[parts.length - 1]);
}

// ============================================================
// Consensus table
// ============================================================

function renderConsensusTable(container, family) {
    const rows = family.members.map(m => `
        <tr>
            <td><a href="/feature/${m.feature_id}">${m.feature_id}</a></td>
            <td style="font-family:monospace">${m.consensus}</td>
            <td>${m.motif_width ?? "\u2014"}</td>
            <td>${m.motif_e_value !== null && m.motif_e_value !== undefined ? Number(m.motif_e_value).toExponential(1) : "\u2014"}</td>
            <td>${fmtVal(m.motif_pr_auc)}</td>
            <td>${fmtVal(m.motif_best_f1)}</td>
            <td>${fmtVal(m.geom_pr_auc)}</td>
            <td style="font-family:monospace;font-size:0.85rem">${m.top_geometric_feature}</td>
            <td>${m.structural_category || "\u2014"}</td>
        </tr>
    `).join("");

    container.innerHTML = `
        <table role="grid" style="font-size:0.85rem;">
            <thead>
                <tr>
                    <th>Node</th>
                    <th>Consensus</th>
                    <th>Width</th>
                    <th>E-value</th>
                    <th>MEME PR-AUC</th>
                    <th>MEME F1</th>
                    <th>Geom PR-AUC</th>
                    <th>Top Geom Feature</th>
                    <th>Structural Category</th>
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}

// ============================================================
// Geometry Feature Importance Heatmap
// ============================================================

function renderHeatmap(container, family, geomFeatureNames) {
    const members = family.members;

    const featureMax = {};
    for (const fn of geomFeatureNames) {
        featureMax[fn] = Math.max(...members.map(m => (m.feature_importances || {})[fn] || 0));
    }
    const activeFeatures = geomFeatureNames.filter(fn => featureMax[fn] > 0.02);

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
        xaxis: { tickangle: -45, tickfont: { size: 10 } },
        yaxis: { tickfont: { size: 11 }, autorange: "reversed" },
    };

    Plotly.newPlot(container, [trace], layout, { responsive: true, displayModeBar: false });
}

// ============================================================
// Per-Node Comparison
// ============================================================

function renderNodeComparison(container, member, featureData, interproData, geometryData) {
    const section = document.createElement("div");
    section.style.marginBottom = "2rem";
    section.style.padding = "1rem";
    section.style.border = "1px solid var(--pico-muted-border-color)";
    section.style.borderRadius = "8px";

    const header = document.createElement("div");
    header.style.marginBottom = "0.75rem";
    header.innerHTML = `
        <div style="display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;">
            <a href="/feature/${member.feature_id}" style="font-size:1.2rem;font-weight:700;">Node ${member.feature_id}</a>
            <span id="radar-node-${member.feature_id}" style="flex-shrink:0;"></span>
            <span class="badge badge-done" style="font-family:monospace">${member.consensus}</span>
            <span class="badge badge-done">MEME PR-AUC: ${fmtVal(member.motif_pr_auc)}</span>
            <span class="badge badge-done">Geom PR-AUC: ${fmtVal(member.geom_pr_auc)}</span>
            <span class="badge badge-count" style="font-family:monospace">${member.top_geometric_feature}</span>
            <span class="badge badge-count">${member.structural_category}</span>
        </div>
    `;
    section.appendChild(header);

    if (member.feature_importances && typeof aggregateToCategories === "function") {
        const scores = aggregateToCategories(member.feature_importances);
        if (scores) {
            const radarSpan = header.querySelector(`#radar-node-${member.feature_id}`);
            renderRadarGlyph(radarSpan, scores, { size: 52, showLabels: false });
        }
    }

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

    const topSeqs = (featureData.top_sequences || []).slice(0, 3);
    const featureMaxAct = featureData.max_activation || 1;

    // Optional InterPro overlay for context
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
        createSequenceStrip(stripDiv, {
            sequence: protein.sequence || "",
            activations: protein.per_residue_activations || [],
            maxActivation: featureMaxAct,
            accession: protein.accession,
            bestAnnotationName: bestAnnotationName,
        });

        section.appendChild(entry);
    }

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
// Cross-Node Activation Overlay
// ============================================================

function renderCrossNodeOverlay(container, memberData) {
    const proteinNodes = {};
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

    const shared = Object.entries(proteinNodes)
        .filter(([, nodes]) => nodes.length >= 2)
        .sort((a, b) => b[1].length - a[1].length);

    if (shared.length === 0) {
        renderFallbackOverlay(container, memberData);
        return;
    }

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
// Main
// ============================================================

document.addEventListener("DOMContentLoaded", async () => {
    const consensus = getConsensusFromUrl();
    if (!consensus) {
        document.getElementById("page-title").textContent = "Error: No consensus in URL";
        return;
    }

    try {
        const res = await fetch("/api/meme-case-study-families");
        if (!res.ok) throw new Error(`Failed to fetch families: ${res.status}`);
        const data = await res.json();

        const family = data.families.find(f => f.representative_consensus === consensus);
        if (!family) throw new Error(`Family ${consensus} not found`);

        document.getElementById("page-title").textContent = family.representative_consensus;
        document.title = `${family.representative_consensus} \u2014 MEME Case Study`;

        document.getElementById("family-info").innerHTML = `
            <strong style="font-family:monospace">${family.representative_consensus}</strong> &mdash;
            ${family.n_nodes} SAE nodes share this MEME motif (representative consensus).
            Mean geometry cosine similarity: <strong>${fmtVal(family.mean_cosine_similarity, 3)}</strong>
            (lower = more diverse geometry).
            ${family.geom_diverse
                ? "Nodes use <strong>different</strong> top geometric features."
                : "Nodes share the <strong>same</strong> top geometric feature."}
        `;

        renderConsensusTable(document.getElementById("consensus-container"), family);

        renderHeatmap(
            document.getElementById("heatmap-container"),
            family,
            data.geometry_feature_names || []
        );

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

        const nodesContainer = document.getElementById("nodes-container");
        nodesContainer.innerHTML = "";
        for (const { member, featureData, interproData, geometryData } of memberData) {
            renderNodeComparison(nodesContainer, member, featureData, interproData, geometryData);
        }

        renderCrossNodeOverlay(
            document.getElementById("overlay-container"),
            memberData
        );

    } catch (err) {
        console.error("Failed to load MEME case study:", err);
        document.getElementById("nodes-container").innerHTML =
            `<p style="color:red">Error: ${err.message}</p>`;
    }
});
