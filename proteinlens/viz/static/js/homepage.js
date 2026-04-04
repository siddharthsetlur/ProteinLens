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
            field: "motif_best_f1",
            headerName: "Motif F1",
            width: 110,
            valueFormatter: nullFormatter(3),
            cellStyle: greenScale(1.0),
            comparator: nullBottomComparator,
            filter: "agNumberColumnFilter",
        },
        {
            field: "motif_best_name",
            headerName: "Best Motif",
            width: 110,
            filter: "agTextColumnFilter",
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
            field: "geometry_residue_gbm_auc_cv",
            headerName: "Geom. GBM AUC",
            width: 140,
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
        // Performance: row virtualization is on by default in AG Grid
        animateRows: false,
        suppressCellFocus: true,
    };

    // AG Grid Community v32 uses createGrid
    agGrid.createGrid(gridDiv, gridOptions);
}

// ============================================================
// Scatter plots: Geom GBM AUC vs InterPro F1
// ============================================================

/**
 * Render two scatter plots from the feature index data:
 *   1. Geom GBM AUC vs InterPro Protein F1
 *   2. Geom GBM AUC vs InterPro Residue F1
 *
 * Only features with both values non-null are plotted.
 * Points are clickable and navigate to the feature detail page.
 *
 * @param {Array} index - Feature index rows from /api/index.
 */
function renderScatterPlots(index) {
    const layout = {
        margin: { t: 40, r: 20, b: 50, l: 60 },
        hovermode: "closest",
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
    };

    const config = { responsive: true, displayModeBar: false };

    // --- Plot 1: AUC vs Protein F1 ---
    const protRows = index.filter(
        (r) => r.geometry_residue_gbm_auc_cv != null && r.interpro_protein_best_f1 != null
    );
    Plotly.newPlot(
        "scatter-protein",
        [{
            x: protRows.map((r) => r.geometry_residue_gbm_auc_cv),
            y: protRows.map((r) => r.interpro_protein_best_f1),
            text: protRows.map((r) => `Feature ${r.feature_id}`),
            customdata: protRows.map((r) => r.feature_id),
            mode: "markers",
            type: "scatter",
            marker: { size: 5, color: "#4361ee", opacity: 0.6 },
        }],
        {
            ...layout,
            title: "Geom GBM AUC vs InterPro Protein F1",
            xaxis: { title: "Geometry GBM AUC (CV)" },
            yaxis: { title: "InterPro Protein F1" },
        },
        config
    );

    // --- Plot 2: AUC vs Residue F1 ---
    const resRows = index.filter(
        (r) => r.geometry_residue_gbm_auc_cv != null && r.interpro_residue_best_f1 != null
    );
    Plotly.newPlot(
        "scatter-residue",
        [{
            x: resRows.map((r) => r.geometry_residue_gbm_auc_cv),
            y: resRows.map((r) => r.interpro_residue_best_f1),
            text: resRows.map((r) => `Feature ${r.feature_id}`),
            customdata: resRows.map((r) => r.feature_id),
            mode: "markers",
            type: "scatter",
            marker: { size: 5, color: "#e63946", opacity: 0.6 },
        }],
        {
            ...layout,
            title: "Geom GBM AUC vs InterPro Residue F1",
            xaxis: { title: "Geometry GBM AUC (CV)" },
            yaxis: { title: "InterPro Residue F1" },
        },
        config
    );

    // Click on point -> navigate to feature page
    for (const divId of ["scatter-protein", "scatter-residue"]) {
        document.getElementById(divId).on("plotly_click", (data) => {
            const fid = data.points[0].customdata;
            window.location.href = `/feature/${fid}`;
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
