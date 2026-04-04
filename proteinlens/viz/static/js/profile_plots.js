/**
 * profile_plots.js — Plotly charts for geometry enrichment visualization.
 *
 * Renders three types of visualizations for each protein in the geometry data:
 *   1. Dual-axis activation vs geometry probability overlay (Plotly)
 *   2. Concordance strip (canvas: agree=green, fp=orange, fn=purple, tn=gray)
 *   3. Top feature trace overlays (Plotly, one trace per geometric feature)
 *
 * Also renders the motif superposition 3D viewer (via mol_viewer.js).
 *
 * Input: geometry enrichment JSON from /api/feature/{id}/geometry,
 *        specifically the plot_data.top_proteins array.
 *
 * Dependencies: Plotly.js (CDN), mol_viewer.js (for motif viewer)
 */

/**
 * Render all geometry plots for a feature into a container.
 *
 * Creates per-protein dual-axis charts, concordance strips, and feature traces,
 * plus the motif superposition viewer if available.
 *
 * @param {HTMLElement} container    - DOM element to append all geometry plots to.
 * @param {Object} geometryData     - Full geometry enrichment JSON from the API.
 */
function renderGeometryPlots(container, geometryData) {
    const plotData = geometryData.plot_data;
    if (!plotData || !plotData.top_proteins || plotData.top_proteins.length === 0) {
        container.innerHTML = '<p class="secondary">No geometry plot data available.</p>';
        return;
    }

    // --- Per-protein plots ---
    for (const protein of plotData.top_proteins) {
        const section = document.createElement("div");
        section.className = "protein-entry";

        // Label
        const label = document.createElement("div");
        label.className = "protein-label";
        label.textContent = `${protein.accession} (${protein.sequence?.length || "?"} residues)`;
        section.appendChild(label);

        // 1. Dual-axis activation vs geometry overlay
        const overlayDiv = document.createElement("div");
        overlayDiv.className = "plot-container";
        section.appendChild(overlayDiv);
        renderDualAxisPlot(overlayDiv, protein);

        // 2. Concordance strip
        if (protein.concordance_labels && protein.concordance_labels.length > 0) {
            const concordDiv = document.createElement("div");
            concordDiv.style.marginBottom = "0.75rem";
            section.appendChild(concordDiv);
            renderConcordanceStrip(concordDiv, protein.concordance_labels);
        }

        // 3. Top feature traces
        if (protein.top_feature_traces && Object.keys(protein.top_feature_traces).length > 0) {
            const tracesDiv = document.createElement("div");
            tracesDiv.className = "plot-container";
            section.appendChild(tracesDiv);
            renderFeatureTraces(tracesDiv, protein);
        }

        container.appendChild(section);
    }

    // --- Motif superposition viewer ---
    const motif = geometryData.geometric_residue_level?.motif_superposition;
    if (motif && motif.mean_structure_pdb) {
        const motifSection = document.createElement("div");
        motifSection.style.marginTop = "1.5rem";

        const motifLabel = document.createElement("h3");
        motifLabel.textContent = "Motif Superposition";
        motifSection.appendChild(motifLabel);

        // Stats
        const motifLen = (motif.per_position_flexibility || []).length;
        const normRmsd = motifLen > 0 ? (motif.mean_rmsd / motifLen).toFixed(3) : "\u2014";
        const statsP = document.createElement("p");
        statsP.className = "secondary";
        statsP.style.fontSize = "0.85rem";
        statsP.textContent = `RMSD/pos: ${normRmsd} \u00c5 ` +
            `(raw: ${motif.mean_rmsd?.toFixed(2) ?? "\u2014"} \u00c5 over ${motifLen} positions), ` +
            `${motif.n_fragments ?? "?"} fragments, ` +
            `Std: ${motif.std_rmsd?.toFixed(2) ?? "\u2014"} \u00c5`;
        motifSection.appendChild(statsP);

        const motifViewer = document.createElement("div");
        motifViewer.className = "viewer-container";
        motifViewer.style.width = "500px";
        motifViewer.style.height = "400px";
        motifSection.appendChild(motifViewer);

        container.appendChild(motifSection);

        // Create the viewer with flexibility coloring
        const flexibility = motif.per_position_flexibility || [];
        createMotifViewer(motifViewer, motif.mean_structure_pdb, flexibility);

        // Per-position flexibility bar chart
        if (flexibility.length > 0) {
            const flexDiv = document.createElement("div");
            flexDiv.className = "plot-container";
            flexDiv.style.marginTop = "1rem";
            motifSection.appendChild(flexDiv);
            renderFlexibilityChart(flexDiv, flexibility);
        }

        // Motif-on-protein overlay: protein selector + 3D viewer
        renderMotifOverlaySelector(
            motifSection, plotData.top_proteins, motif, geometryData.feature_max_activation || 1
        );
    }
}

/**
 * Render a dropdown selector for overlaying the motif onto a chosen protein.
 *
 * For each protein, the peak activated position is found and a Kabsch-aligned
 * motif is superimposed in a 3D viewer alongside the full protein structure.
 *
 * @param {HTMLElement} parent       - Container to append the selector + viewer to.
 * @param {Array<Object>} proteins   - plot_data.top_proteins array.
 * @param {Object} motif             - motif_superposition data.
 * @param {number} featureMaxAct     - Feature-level max activation.
 */
function renderMotifOverlaySelector(parent, proteins, motif, featureMaxAct) {
    const motifLength = (motif.per_position_flexibility || []).length;
    if (!motif.mean_structure_pdb || motifLength === 0) return;

    // Filter to proteins that have ca_backbone and activated_positions
    const eligible = proteins.filter(p =>
        p.ca_backbone && p.ca_backbone.length > 0 &&
        p.activated_positions && p.activated_positions.length > 0
    );
    if (eligible.length === 0) return;

    const halfW = Math.floor((motifLength - 1) / 2);

    // Build options with peak position info
    const options = eligible.map(p => {
        const peak = p.activated_positions.reduce(
            (a, b) => (b.activation > a.activation ? b : a),
            p.activated_positions[0]
        );
        return { protein: p, peakPos: peak.position, peakAct: peak.activation };
    }).filter(o => o.peakPos >= halfW && o.peakPos + halfW < o.protein.ca_backbone.length);

    if (options.length === 0) return;

    // --- UI ---
    const wrapper = document.createElement("div");
    wrapper.style.marginTop = "1.5rem";

    const heading = document.createElement("h4");
    heading.textContent = "Motif Overlay on Protein";
    wrapper.appendChild(heading);

    const desc = document.createElement("p");
    desc.className = "secondary";
    desc.style.fontSize = "0.85rem";
    desc.textContent = "Select a protein to see the motif (green) Kabsch-aligned onto its structure at the peak activation site.";
    wrapper.appendChild(desc);

    const select = document.createElement("select");
    select.style.marginBottom = "0.75rem";
    select.style.maxWidth = "400px";
    for (const opt of options) {
        const el = document.createElement("option");
        el.value = opt.protein.accession;
        el.textContent = `${opt.protein.accession} (peak res ${opt.peakPos + 1}, act ${opt.peakAct.toFixed(3)})`;
        select.appendChild(el);
    }
    wrapper.appendChild(select);

    const viewerDiv = document.createElement("div");
    viewerDiv.className = "viewer-container";
    viewerDiv.style.width = "600px";
    viewerDiv.style.height = "450px";
    wrapper.appendChild(viewerDiv);

    parent.appendChild(wrapper);

    // Load overlay for the selected protein
    function loadOverlay(accession) {
        const opt = options.find(o => o.protein.accession === accession);
        if (!opt) return;

        // Clear any previous RMSD label
        const oldLabel = viewerDiv.nextElementSibling;
        if (oldLabel && oldLabel.classList.contains("secondary")) {
            oldLabel.remove();
        }

        const p = opt.protein;
        const activations = p.sae_activation_profile || [];

        createMotifOverlayViewer(
            viewerDiv, p.accession, activations, featureMaxAct,
            p.ca_backbone, opt.peakPos, motif.mean_structure_pdb, motifLength
        );
    }

    select.addEventListener("change", () => loadOverlay(select.value));

    // Load first protein immediately
    loadOverlay(options[0].protein.accession);
}

/**
 * Render a bar chart of per-position flexibility (std of aligned coords).
 *
 * Blue (rigid) -> Red (flexible), matching the 3D viewer coloring.
 *
 * @param {HTMLElement} div          - DOM element for the Plotly chart.
 * @param {Array<number>} flexibility - Per-position flexibility values (Angstroms).
 */
function renderFlexibilityChart(div, flexibility) {
    const maxFlex = Math.max(...flexibility, 0.01);
    const xPositions = flexibility.map((_, i) => i + 1);

    // Color each bar blue->red matching the 3D viewer
    const colors = flexibility.map((v) => {
        const norm = Math.min(v / maxFlex, 1);
        const r = Math.round(norm * 255);
        const b = Math.round((1 - norm) * 255);
        return `rgb(${r},0,${b})`;
    });

    const traces = [{
        x: xPositions,
        y: flexibility,
        type: "bar",
        marker: { color: colors },
        hovertemplate: "Position %{x}<br>Flexibility: %{y:.3f} \u00c5<extra></extra>",
    }];

    const layout = {
        title: { text: "Per-Position Flexibility (\u00c5)", font: { size: 13 } },
        xaxis: { title: "Motif Position" },
        yaxis: { title: "Std of Aligned Coords (\u00c5)" },
        height: 250,
        margin: { t: 40, b: 40, l: 60, r: 20 },
    };

    Plotly.newPlot(div, traces, layout, { responsive: true, displayModeBar: false });
}

/**
 * Render a dual-axis Plotly chart: SAE activation (red, left Y) vs
 * geometry probability (blue, right Y) along the sequence.
 *
 * Shaded regions mark activated positions.
 *
 * @param {HTMLElement} div      - DOM element for the Plotly chart.
 * @param {Object} protein       - Per-protein data from plot_data.top_proteins.
 */
function renderDualAxisPlot(div, protein) {
    const seqLen = protein.sae_activation_profile?.length || 0;
    if (seqLen === 0) return;

    // X axis: 1-based residue positions
    const xPositions = Array.from({ length: seqLen }, (_, i) => i + 1);

    const traces = [
        {
            x: xPositions,
            y: protein.sae_activation_profile,
            name: "SAE Activation",
            type: "scatter",
            mode: "lines",
            line: { color: "#dc2626", width: 1.5 },
            yaxis: "y",
        },
        {
            x: xPositions,
            y: protein.geom_prob_profile,
            name: "Geom. Probability",
            type: "scatter",
            mode: "lines",
            line: { color: "#2563eb", width: 1.5 },
            yaxis: "y2",
        },
    ];

    // Add shaded regions for activated positions
    const shapes = [];
    if (protein.activated_positions) {
        for (const pos of protein.activated_positions) {
            shapes.push({
                type: "rect",
                xref: "x",
                yref: "paper",
                x0: pos.position - 0.5,
                x1: pos.position + 0.5,
                y0: 0,
                y1: 1,
                fillcolor: "rgba(220,38,38,0.1)",
                line: { width: 0 },
            });
        }
    }

    const layout = {
        title: { text: "Activation vs Geometry", font: { size: 13 } },
        xaxis: { title: "Residue Position" },
        yaxis: {
            title: { text: "SAE Activation", font: { color: "#dc2626" } },
            side: "left",
        },
        yaxis2: {
            title: { text: "Geom. Probability", font: { color: "#2563eb" } },
            side: "right",
            overlaying: "y",
        },
        shapes: shapes,
        height: 280,
        margin: { t: 40, b: 40, l: 60, r: 60 },
        legend: { x: 0.01, y: 0.99, bgcolor: "rgba(255,255,255,0.8)" },
    };

    Plotly.newPlot(div, traces, layout, { responsive: true, displayModeBar: false });
}

/**
 * Render a concordance canvas strip: each residue gets a color based on
 * its concordance label.
 *
 *   agree = green (#28a745)
 *   fp    = orange (#fd7e14)
 *   fn    = purple (#6f42c1)
 *   tn    = gray (#ccc)
 *
 * @param {HTMLElement} container       - DOM element to append the strip + legend to.
 * @param {Array<string>} labels        - Per-residue concordance labels.
 */
function renderConcordanceStrip(container, labels) {
    // Color legend
    const legend = document.createElement("div");
    legend.className = "concordance-legend";
    legend.innerHTML = `
        <span class="legend-agree">Agree</span>
        <span class="legend-fp">FP</span>
        <span class="legend-fn">FN</span>
        <span class="legend-tn">TN</span>
    `;
    container.appendChild(legend);

    // Canvas strip
    const canvas = document.createElement("canvas");
    const stripHeight = 20;
    canvas.style.width = "100%";
    canvas.style.height = stripHeight + "px";
    container.appendChild(canvas);

    const colorLookup = {
        agree: "#28a745",
        fp: "#fd7e14",
        fn: "#6f42c1",
        tn: "#cccccc",
    };

    requestAnimationFrame(() => {
        const displayWidth = container.clientWidth;
        if (displayWidth === 0) return;

        const dpr = window.devicePixelRatio || 1;
        canvas.width = displayWidth * dpr;
        canvas.height = stripHeight * dpr;
        const ctx = canvas.getContext("2d");
        ctx.scale(dpr, dpr);

        const colWidth = displayWidth / labels.length;
        for (let i = 0; i < labels.length; i++) {
            ctx.fillStyle = colorLookup[labels[i]] || "#cccccc";
            ctx.fillRect(i * colWidth, 0, colWidth + 0.5, stripHeight);
        }
    });
}

/**
 * Render top geometric feature traces as a Plotly multi-line chart.
 *
 * Each trace is a different geometric feature (e.g., curvature, contact density)
 * plotted along the sequence. Null values in the trace indicate positions
 * outside the feature's computation window.
 *
 * @param {HTMLElement} div      - DOM element for the Plotly chart.
 * @param {Object} protein       - Per-protein data with top_feature_traces dict.
 */
function renderFeatureTraces(div, protein) {
    const traces = [];
    const traceData = protein.top_feature_traces;
    const seqLen = Object.values(traceData)[0]?.length || 0;
    const xPositions = Array.from({ length: seqLen }, (_, i) => i + 1);

    // Cycle through a palette for distinct trace colors
    const palette = [
        "#dc2626", "#2563eb", "#16a34a", "#d97706", "#7c3aed",
        "#db2777", "#0891b2", "#65a30d",
    ];
    let colorIdx = 0;

    for (const [featureName, values] of Object.entries(traceData)) {
        traces.push({
            x: xPositions,
            y: values,
            name: featureName,
            type: "scatter",
            mode: "lines",
            line: { color: palette[colorIdx % palette.length], width: 1.5 },
            connectgaps: false, // Don't connect across null gaps
        });
        colorIdx++;
    }

    const layout = {
        title: { text: "Top Geometric Feature Profiles", font: { size: 13 } },
        xaxis: { title: "Residue Position" },
        yaxis: { title: "Feature Value" },
        height: 280,
        margin: { t: 40, b: 40, l: 60, r: 20 },
        legend: { x: 0.01, y: 0.99, bgcolor: "rgba(255,255,255,0.8)" },
    };

    Plotly.newPlot(div, traces, layout, { responsive: true, displayModeBar: false });
}
