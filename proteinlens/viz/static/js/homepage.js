/**
 * homepage.js — AG Grid setup, stats rendering, and row-click navigation.
 *
 * On page load:
 *   1. Fetches /api/stats and /api/index in parallel
 *   2. Renders model card, dataset card, and pipeline status badges
 *   3. Initializes AG Grid with sortable/filterable columns and color-coded cells
 *   4. Row click navigates to /feature/{feature_id}
 *
 * External dependencies (loaded via CDN in index.html):
 *   - AG Grid Community (agGrid global)
 */

// ============================================================
// Utility: format a number for display, handling null
// ============================================================

/**
 * Format a numeric value for display in the table or cards.
 *
 * @param {number|null} val   - The value to format.
 * @param {number} decimals   - Decimal places to show (default 3).
 * @returns {string}          - Formatted string, or "—" for null/undefined.
 */
function fmt(val, decimals = 3) {
    if (val === null || val === undefined) return "—";
    return Number(val).toFixed(decimals);
}

// ============================================================
// Card rendering
// ============================================================

/**
 * Render the SAE model card from /api/stats response.
 *
 * Displays architecture, dictionary size, expansion factor, activation dim,
 * L1 penalty, learning rate, training steps, and wandb name.
 *
 * @param {Object} sae - The stats.sae object from the API.
 */
function renderSaeCard(sae) {
    const body = document.getElementById("sae-card-body");
    if (!sae || Object.keys(sae).length === 0) {
        body.textContent = "SAE config not found";
        return;
    }
    body.innerHTML = `<dl>
        <dt>Architecture</dt><dd>${sae.architecture || "—"}</dd>
        <dt>Dict Size</dt><dd>${sae.dictionary_size ?? "—"}</dd>
        <dt>Expansion</dt><dd>${sae.expansion_factor ?? "—"}x</dd>
        <dt>Activation Dim</dt><dd>${sae.activation_dim ?? "—"}</dd>
        <dt>L1 Penalty</dt><dd>${fmt(sae.l1_penalty, 5)}</dd>
        <dt>Learning Rate</dt><dd>${fmt(sae.lr, 7)}</dd>
        <dt>Steps</dt><dd>${sae.steps?.toLocaleString() ?? "—"}</dd>
        <dt>Wandb</dt><dd>${sae.wandb_name || "—"}</dd>
    </dl>`;
}

/**
 * Render the dataset card from /api/stats response.
 *
 * Shows ESM model, layer, organism, protein/cluster counts, and threshold.
 *
 * @param {Object} dataset - The stats.dataset object from the API.
 */
function renderDatasetCard(dataset) {
    const body = document.getElementById("dataset-card-body");
    if (!dataset || Object.keys(dataset).length === 0) {
        body.textContent = "Dataset stats not found";
        return;
    }
    body.innerHTML = `<dl>
        <dt>ESM Model</dt><dd>${dataset.esm_model || "—"}</dd>
        <dt>Layer</dt><dd>${dataset.esm_layer ?? "—"}</dd>
        <dt>Organism</dt><dd>Taxon ${dataset.organism_taxid ?? "—"}</dd>
        <dt>Proteins</dt><dd>${dataset.total_proteins?.toLocaleString() ?? "—"}</dd>
        <dt>Clusters</dt><dd>${dataset.total_clusters?.toLocaleString() ?? "—"}</dd>
        <dt>Threshold</dt><dd>${dataset.activation_threshold ?? "—"}</dd>
        <dt>Features</dt><dd>${dataset.num_features?.toLocaleString() ?? "—"}</dd>
    </dl>`;
}

/**
 * Render pipeline status badges showing completion counts.
 *
 * Displays feature/interpro/geometry file counts as badges, plus a list
 * of completed pipeline stages.
 *
 * @param {Object} pipeline - The stats.pipeline object from the API.
 * @param {number} numFeatures - Total number of features (for X/total display).
 */
function renderPipelineCard(pipeline, numFeatures) {
    const body = document.getElementById("pipeline-card-body");
    if (!pipeline) {
        body.textContent = "Pipeline status not available";
        return;
    }

    const total = numFeatures || "?";

    // Build count badges
    const counts = [
        { label: "Features", count: pipeline.feature_count },
        { label: "InterPro", count: pipeline.interpro_count },
        { label: "Geometry", count: pipeline.geometry_count },
        { label: "Motif", count: pipeline.motif_count },
    ];

    const badgeHtml = counts
        .map((c) => {
            const isDone = c.count >= total;
            const cls = isDone ? "badge badge-done" : "badge badge-count";
            return `<span class="${cls}">${c.label}: ${c.count}/${total}</span>`;
        })
        .join(" ");

    // Build completed stages list
    const stages = pipeline.completed_stages || [];
    const stageHtml = stages
        .map((s) => `<span class="badge badge-done">${s}</span>`)
        .join(" ");

    body.innerHTML = `
        <div style="margin-bottom:0.5rem">${badgeHtml}</div>
        <div style="font-size:0.8rem; line-height:1.8">${stageHtml}</div>
    `;
}

// ============================================================
// AG Grid setup
// ============================================================

/**
 * Return a cellStyle function that applies green intensity proportional
 * to a value between 0 and maxVal.
 *
 * Null/undefined values get no special styling; they're handled by the
 * valueFormatter which shows "—".
 *
 * @param {number} maxVal - The value at which color is fully saturated green.
 * @returns {Function} AG Grid cellStyle callback.
 */
function greenScale(maxVal) {
    return (params) => {
        const v = params.value;
        if (v === null || v === undefined) return null;
        // Clamp between 0 and 1
        const intensity = Math.min(Math.max(v / maxVal, 0), 1);
        // Interpolate: white (255,255,255) -> green (40,167,69)
        const r = Math.round(255 - intensity * (255 - 40));
        const g = Math.round(255 - intensity * (255 - 167));
        const b = Math.round(255 - intensity * (255 - 69));
        return { backgroundColor: `rgb(${r},${g},${b})` };
    };
}

/**
 * Null-safe value formatter: shows "—" for null, otherwise fixed decimals.
 *
 * @param {number} decimals - Number of decimal places.
 * @returns {Function} AG Grid valueFormatter callback.
 */
function nullFormatter(decimals = 3) {
    return (params) => {
        if (params.value === null || params.value === undefined) return "—";
        return Number(params.value).toFixed(decimals);
    };
}

/**
 * Custom comparator that sorts null values to the bottom regardless
 * of sort direction.
 *
 * AG Grid v32 passes (valueA, valueB, nodeA, nodeB, isDescending) to
 * custom comparators. We flip the null-handling when descending so that
 * nulls always stay at the bottom of the table.
 *
 * @param {*} a - First value.
 * @param {*} b - Second value.
 * @param {*} _nodeA - AG Grid row node (unused).
 * @param {*} _nodeB - AG Grid row node (unused).
 * @param {boolean} isDescending - Whether the column is sorted descending.
 * @returns {number} Comparison result.
 */
function nullBottomComparator(a, b, _nodeA, _nodeB, isDescending) {
    if (a === null && b === null) return 0;
    if (a === null) return isDescending ? -1 : 1;
    if (b === null) return isDescending ? 1 : -1;
    return a - b;
}

/**
 * Build AG Grid column definitions for the feature table.
 *
 * All columns are sortable and filterable. Numeric columns use null-safe
 * formatters and green color scales. Row click navigates to the feature page.
 *
 * @returns {Array} AG Grid column definitions.
 */
function buildColumnDefs() {
    return [
        {
            field: "feature_id",
            headerName: "ID",
            width: 80,
            filter: "agNumberColumnFilter",
        },
        {
            field: "max_activation",
            headerName: "Max Act.",
            width: 110,
            valueFormatter: nullFormatter(4),
            comparator: nullBottomComparator,
            filter: "agNumberColumnFilter",
        },
        {
            field: "pct_proteins_activated",
            headerName: "% Proteins",
            width: 115,
            valueFormatter: nullFormatter(1),
            comparator: nullBottomComparator,
            filter: "agNumberColumnFilter",
        },
        {
            field: "pct_clusters_activated",
            headerName: "% Clusters",
            width: 115,
            valueFormatter: nullFormatter(1),
            comparator: nullBottomComparator,
            filter: "agNumberColumnFilter",
        },
        {
            field: "interpro_protein_best_f1",
            headerName: "InterPro Prot. F1",
            width: 140,
            valueFormatter: nullFormatter(3),
            cellStyle: greenScale(1.0),
            comparator: nullBottomComparator,
            filter: "agNumberColumnFilter",
        },
        {
            field: "interpro_protein_best_name",
            headerName: "InterPro Annotation",
            width: 200,
            valueFormatter: (params) => params.value || "—",
            filter: "agTextColumnFilter",
        },
        {
            field: "interpro_residue_best_f1",
            headerName: "InterPro Res. F1",
            width: 140,
            valueFormatter: nullFormatter(3),
            cellStyle: greenScale(1.0),
            comparator: nullBottomComparator,
            filter: "agNumberColumnFilter",
        },
        {
            field: "motif_best_pr_auc",
            headerName: "Motif PR-AUC",
            width: 120,
            valueFormatter: nullFormatter(3),
            cellStyle: greenScale(1.0),
            comparator: nullBottomComparator,
            filter: "agNumberColumnFilter",
        },
        {
            field: "motif_best_consensus",
            headerName: "Best Motif",
            width: 130,
            filter: "agTextColumnFilter",
            cellStyle: () => ({ fontFamily: "monospace" }),
        },
        {
            field: "position_best_f1",
            headerName: "Position F1",
            width: 110,
            valueFormatter: nullFormatter(3),
            cellStyle: greenScale(1.0),
            comparator: nullBottomComparator,
            filter: "agNumberColumnFilter",
        },
        {
            field: "position_best_name",
            headerName: "Best Position",
            width: 130,
            valueFormatter: (params) => params.value || "\u2014",
            filter: "agTextColumnFilter",
        },
        {
            field: "cath_best_f1",
            headerName: "CATH F1",
            width: 110,
            valueFormatter: nullFormatter(3),
            cellStyle: greenScale(1.0),
            comparator: nullBottomComparator,
            filter: "agNumberColumnFilter",
        },
        {
            field: "geometry_radar",
            headerName: "Geom. Profile",
            width: 80,
            sortable: false,
            filter: false,
            cellRenderer: (params) => {
                const radar = params.value;
                if (!radar) return "";
                const div = document.createElement("div");
                div.style.cssText = "display:flex;align-items:center;justify-content:center;height:100%;";
                const scores = [
                    radar.curvature || 0, radar.torsion || 0,
                    radar.planarity || 0, radar.compactness || 0,
                    radar.contacts || 0, radar.composition || 0,
                ];
                renderRadarGlyph(div, scores, { size: 48, showLabels: false });
                return div;
            },
        },
        {
            field: "geometry_protein_r2_cv",
            headerName: "Geom. R2 CV",
            width: 130,
            valueFormatter: nullFormatter(3),
            cellStyle: greenScale(1.0),
            comparator: nullBottomComparator,
            filter: "agNumberColumnFilter",
        },
        {
            field: "geometry_residue_pr_auc",
            headerName: "Geom. PR-AUC",
            width: 140,
            valueFormatter: nullFormatter(3),
            cellStyle: greenScale(1.0),
            comparator: nullBottomComparator,
            filter: "agNumberColumnFilter",
        },
        {
            field: "is_geometry_primary",
            headerName: "Geom. Primary",
            width: 80,
            valueFormatter: (params) => params.value === true ? "\u2713" : "",
            cellStyle: (params) => params.value === true ? { color: "#28a745", fontWeight: "bold", textAlign: "center" } : { textAlign: "center" },
            filter: "agTextColumnFilter",
        },
        {
            field: "geometry_primary_score",
            headerName: "Geom. Score",
            width: 110,
            valueFormatter: nullFormatter(3),
            cellStyle: greenScale(1.0),
            comparator: nullBottomComparator,
            filter: "agNumberColumnFilter",
        },
    ];
}

/**
 * Initialize the AG Grid with feature index data.
 *
 * Sets up row virtualization, sorting, filtering, and row-click navigation.
 *
 * @param {Array} rowData - Array of feature row objects from /api/index.
 */
function initGrid(rowData) {
    const gridDiv = document.getElementById("feature-grid");

    const gridOptions = {
        columnDefs: buildColumnDefs(),
        rowData: rowData,
        defaultColDef: {
            sortable: true,
            resizable: true,
        },
        // Row click -> navigate to feature detail page
        onRowClicked: (event) => {
            const featureId = event.data.feature_id;
            window.location.href = `/feature/${featureId}`;
        },
        // Taller rows to fit radar glyphs
        rowHeight: 56,
        // Performance: row virtualization is on by default in AG Grid
        animateRows: false,
        suppressCellFocus: true,
    };

    // AG Grid Community v32 uses createGrid
    agGrid.createGrid(gridDiv, gridOptions);
}

// ============================================================
// Scatter plots: density-colored, seaborn-styled Plotly plots
// ============================================================

/**
 * Estimate per-point density via fast 2D grid binning.
 * Returns an array of density values (one per point), normalised to [0, 1].
 */
function estimateDensity(xs, ys, nBins = 40) {
    const n = xs.length;
    if (n === 0) return [];

    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;

    // Count points per bin
    const grid = Array.from({ length: nBins }, () => new Float32Array(nBins));
    const binXs = new Int32Array(n);
    const binYs = new Int32Array(n);
    for (let i = 0; i < n; i++) {
        const bx = Math.min(Math.floor((xs[i] - xMin) / xRange * nBins), nBins - 1);
        const by = Math.min(Math.floor((ys[i] - yMin) / yRange * nBins), nBins - 1);
        binXs[i] = bx;
        binYs[i] = by;
        grid[bx][by]++;
    }

    // Gaussian blur (3x3 kernel) for smoothing
    const smoothed = Array.from({ length: nBins }, () => new Float32Array(nBins));
    const k = [0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625];
    for (let i = 0; i < nBins; i++) {
        for (let j = 0; j < nBins; j++) {
            let val = 0;
            let ki = 0;
            for (let di = -1; di <= 1; di++) {
                for (let dj = -1; dj <= 1; dj++) {
                    const ni = i + di, nj = j + dj;
                    if (ni >= 0 && ni < nBins && nj >= 0 && nj < nBins) {
                        val += grid[ni][nj] * k[ki];
                    }
                    ki++;
                }
            }
            smoothed[i][j] = val;
        }
    }

    // Look up each point's density
    const densities = new Float64Array(n);
    let maxD = 0;
    for (let i = 0; i < n; i++) {
        densities[i] = smoothed[binXs[i]][binYs[i]];
        if (densities[i] > maxD) maxD = densities[i];
    }
    // Normalise to [0, 1]
    if (maxD > 0) {
        for (let i = 0; i < n; i++) densities[i] /= maxD;
    }
    return Array.from(densities);
}

/**
 * Create a density-colored scatter plot with seaborn-like styling.
 *
 * @param {string} divId     - Target div id.
 * @param {Array}  rows      - Data rows (each must have x, y, feature_id fields).
 * @param {string} xField    - Key for x values.
 * @param {string} yField    - Key for y values.
 * @param {string} title     - Plot title.
 * @param {string} xlabel    - X-axis label.
 * @param {string} ylabel    - Y-axis label.
 * @param {Array}  colorscale - Plotly colorscale (seaborn-like).
 */
function densityScatter(divId, rows, xField, yField, title, xlabel, ylabel, colorscale) {
    const xs = rows.map((r) => r[xField]);
    const ys = rows.map((r) => r[yField]);
    const ids = rows.map((r) => r.feature_id);
    const density = estimateDensity(xs, ys);

    // Sort by density ascending so dense points render on top
    const indices = density.map((_, i) => i).sort((a, b) => density[a] - density[b]);

    const traces = [
        // Background contour for filled density regions
        {
            x: xs,
            y: ys,
            type: "histogram2dcontour",
            colorscale: colorscale,
            showscale: false,
            contours: { coloring: "fill", showlines: false },
            ncontours: 15,
            opacity: 0.35,
            hoverinfo: "skip",
        },
        // Scatter points colored by density
        {
            x: indices.map((i) => xs[i]),
            y: indices.map((i) => ys[i]),
            text: indices.map((i) => `Feature ${ids[i]}`),
            customdata: indices.map((i) => ids[i]),
            mode: "markers",
            type: "scatter",
            marker: {
                size: 4.5,
                color: indices.map((i) => density[i]),
                colorscale: colorscale,
                showscale: true,
                colorbar: { title: "Density", thickness: 12, len: 0.6, tickfont: { size: 9 } },
                opacity: 0.8,
                line: { width: 0 },
            },
            hovertemplate: "%{text}<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>",
        },
    ];

    const layout = {
        title: { text: title, font: { size: 13, family: "sans-serif" } },
        xaxis: {
            title: { text: xlabel, font: { size: 11 } },
            tickfont: { size: 10 },
            gridcolor: "#e9ecef",
            gridwidth: 1,
            zeroline: false,
        },
        yaxis: {
            title: { text: ylabel, font: { size: 11 } },
            tickfont: { size: 10 },
            gridcolor: "#e9ecef",
            gridwidth: 1,
            zeroline: false,
        },
        margin: { t: 45, r: 60, b: 50, l: 55 },
        hovermode: "closest",
        paper_bgcolor: "#fff",
        plot_bgcolor: "#f8f9fa",
        showlegend: false,
        font: { family: "sans-serif" },
    };

    Plotly.newPlot(divId, traces, layout, { responsive: true, displayModeBar: false });
}

/**
 * Render all scatter plots from the feature index data.
 * Uses density-colored scatter with contour backgrounds.
 *
 * @param {Array} index - Feature index rows from /api/index.
 */
function renderScatterPlots(index) {
    // Seaborn-inspired colorscales
    const csMako    = [[0,"#0b0405"],[0.25,"#35628b"],[0.5,"#3d8e8a"],[0.75,"#8fd0a3"],[1,"#dff8d2"]];
    const csRocket  = [[0,"#03051a"],[0.25,"#6b1d5e"],[0.5,"#cb3b47"],[0.75,"#f0944d"],[1,"#faebdd"]];
    const csViridis = [[0,"#440154"],[0.25,"#3b528b"],[0.5,"#21918c"],[0.75,"#5ec962"],[1,"#fde725"]];
    const csFlare   = [[0,"#e98d6b"],[0.25,"#cc5b6a"],[0.5,"#8f3a84"],[0.75,"#4c2c7a"],[1,"#180e4a"]];
    const csCrest   = [[0,"#1a1530"],[0.25,"#1b6b72"],[0.5,"#36a66d"],[0.75,"#a0d55e"],[1,"#f0f921"]];
    const csIce     = [[0,"#0b0405"],[0.25,"#2a4858"],[0.5,"#4e8a7e"],[0.75,"#8ec8a5"],[1,"#d6f5d6"]];

    const plots = [
        {
            divId: "scatter-protein",
            xField: "geometry_residue_pr_auc",
            yField: "interpro_protein_best_f1",
            title: "Geometry PR-AUC vs InterPro Protein F1",
            xlabel: "Geometry PR-AUC",
            ylabel: "InterPro Protein F1",
            colorscale: csMako,
        },
        {
            divId: "scatter-residue",
            xField: "geometry_residue_pr_auc",
            yField: "interpro_residue_best_f1",
            title: "Geometry PR-AUC vs InterPro Residue F1",
            xlabel: "Geometry PR-AUC",
            ylabel: "InterPro Residue F1",
            colorscale: csRocket,
        },
        {
            divId: "scatter-motif",
            xField: "geometry_residue_pr_auc",
            yField: "motif_best_pr_auc",
            title: "Geometry PR-AUC vs Motif PR-AUC",
            xlabel: "Geometry PR-AUC",
            ylabel: "Motif PR-AUC",
            colorscale: csViridis,
        },
        {
            divId: "scatter-position",
            xField: "geometry_residue_pr_auc",
            yField: "position_best_f1",
            title: "Geometry PR-AUC vs Position F1",
            xlabel: "Geometry PR-AUC",
            ylabel: "Position F1",
            colorscale: csFlare,
        },
        {
            divId: "scatter-cath",
            xField: "geometry_residue_pr_auc",
            yField: "cath_best_f1",
            title: "Geometry PR-AUC vs CATH F1",
            xlabel: "Geometry PR-AUC",
            ylabel: "CATH F1",
            colorscale: csIce,
        },
    ];

    for (const p of plots) {
        const rows = index.filter((r) => r[p.xField] != null && r[p.yField] != null);
        densityScatter(p.divId, rows, p.xField, p.yField, p.title, p.xlabel, p.ylabel, p.colorscale);
    }

    // --- Best F1 plot ---
    const bestRows = index
        .map((r) => {
            const f1s = [
                r.interpro_protein_best_f1,
                r.interpro_residue_best_f1,
                r.motif_best_pr_auc,
                r.position_best_f1,
                r.cath_best_f1,
            ].filter((v) => v != null);
            if (f1s.length === 0 || r.geometry_residue_pr_auc == null) return null;
            return { ...r, best_f1: Math.max(...f1s) };
        })
        .filter((r) => r != null);
    densityScatter(
        "scatter-best-f1", bestRows,
        "geometry_residue_pr_auc", "best_f1",
        "Geometry PR-AUC vs Best F1",
        "Geometry PR-AUC", "Best F1 (max of all metrics)",
        csCrest
    );

    // --- Geometry-primary plot (custom: gold highlights over grey density) ---
    const gpRows = index
        .map((r) => {
            const seqF1s = [r.motif_best_pr_auc, r.position_best_f1, r.interpro_residue_best_f1, r.cath_best_f1].filter((v) => v != null);
            if (r.geometry_residue_pr_auc == null) return null;
            return { ...r, best_seq_f1: seqF1s.length > 0 ? Math.max(...seqF1s) : 0 };
        })
        .filter((r) => r != null);

    const gpPrimary = gpRows.filter((r) => r.is_geometry_primary === true);
    const gpOther = gpRows.filter((r) => r.is_geometry_primary !== true);

    const otherXs = gpOther.map((r) => r.geometry_residue_pr_auc);
    const otherYs = gpOther.map((r) => r.best_seq_f1);
    const otherDens = estimateDensity(otherXs, otherYs);
    const otherIdx = otherDens.map((_, i) => i).sort((a, b) => otherDens[a] - otherDens[b]);

    const csGrey = [[0, "#f8f9fa"], [0.5, "#adb5bd"], [1, "#495057"]];

    Plotly.newPlot(
        "scatter-geom-primary",
        [
            {
                x: otherXs,
                y: otherYs,
                type: "histogram2dcontour",
                colorscale: csGrey,
                showscale: false,
                contours: { coloring: "fill", showlines: false },
                ncontours: 12,
                opacity: 0.3,
                hoverinfo: "skip",
            },
            {
                x: otherIdx.map((i) => otherXs[i]),
                y: otherIdx.map((i) => otherYs[i]),
                text: otherIdx.map((i) => `Feature ${gpOther[i].feature_id}`),
                customdata: otherIdx.map((i) => gpOther[i].feature_id),
                mode: "markers",
                type: "scatter",
                marker: {
                    size: 3.5,
                    color: otherIdx.map((i) => otherDens[i]),
                    colorscale: csGrey,
                    showscale: false,
                    opacity: 0.5,
                    line: { width: 0 },
                },
                name: "Other",
                hovertemplate: "%{text}<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>",
            },
            {
                x: gpPrimary.map((r) => r.geometry_residue_pr_auc),
                y: gpPrimary.map((r) => r.best_seq_f1),
                text: gpPrimary.map((r) => `Feature ${r.feature_id}`),
                customdata: gpPrimary.map((r) => r.feature_id),
                mode: "markers",
                type: "scatter",
                marker: {
                    size: 7,
                    color: "#f59f00",
                    opacity: 0.9,
                    line: { color: "#c77c00", width: 0.5 },
                },
                name: `Geometry-primary (${gpPrimary.length})`,
                hovertemplate: "%{text}<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>",
            },
        ],
        {
            title: { text: `Geometry PR-AUC vs Best Sequence F1 (${gpPrimary.length} geometry-primary)`, font: { size: 13, family: "sans-serif" } },
            xaxis: { title: { text: "Geometry PR-AUC", font: { size: 11 } }, tickfont: { size: 10 }, gridcolor: "#e9ecef", zeroline: false },
            yaxis: { title: { text: "Best Sequence F1", font: { size: 11 } }, tickfont: { size: 10 }, gridcolor: "#e9ecef", zeroline: false },
            margin: { t: 45, r: 20, b: 50, l: 55 },
            hovermode: "closest",
            paper_bgcolor: "#fff",
            plot_bgcolor: "#f8f9fa",
            showlegend: true,
            legend: { x: 0.02, y: 0.98, font: { size: 10 }, bgcolor: "rgba(255,255,255,0.85)" },
            font: { family: "sans-serif" },
            shapes: [
                { type: "line", x0: 0.3, x1: 0.3, y0: 0, y1: 1, line: { color: "#868e96", width: 1, dash: "dash" } },
            ],
        },
        { responsive: true, displayModeBar: false }
    );

    // Click on any point -> navigate to feature page
    const allScatterDivs = ["scatter-protein", "scatter-residue", "scatter-motif", "scatter-position", "scatter-cath", "scatter-best-f1", "scatter-geom-primary"];
    for (const divId of allScatterDivs) {
        document.getElementById(divId).on("plotly_click", (data) => {
            const fid = data.points[0].customdata;
            if (fid != null) window.location.href = `/feature/${fid}`;
        });
    }
}

// ============================================================
// Main: fetch data and render
// ============================================================

document.addEventListener("DOMContentLoaded", async () => {
    try {
        // Fetch stats and index in parallel
        const [statsRes, indexRes] = await Promise.all([
            fetch("/api/stats"),
            fetch("/api/index"),
        ]);

        if (!statsRes.ok) throw new Error(`Stats fetch failed: ${statsRes.status}`);
        if (!indexRes.ok) throw new Error(`Index fetch failed: ${indexRes.status}`);

        const stats = await statsRes.json();
        const index = await indexRes.json();

        // Update subtitle
        const subtitle = document.getElementById("subtitle");
        const ds = stats.dataset || {};
        subtitle.textContent = `${ds.esm_model || "ESM"} layer ${ds.esm_layer ?? "?"} · ${ds.num_features?.toLocaleString() ?? "?"} features · ${ds.total_proteins?.toLocaleString() ?? "?"} proteins`;

        // Render info cards
        renderSaeCard(stats.sae || {});
        renderDatasetCard(stats.dataset || {});
        renderPipelineCard(stats.pipeline || {}, ds.num_features);

        // Initialize feature table
        initGrid(index);

        // Render scatter plots
        renderScatterPlots(index);
    } catch (err) {
        console.error("Failed to load homepage data:", err);
        document.getElementById("subtitle").textContent = `Error: ${err.message}`;
    }
});
