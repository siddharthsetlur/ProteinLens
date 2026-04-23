/**
 * homepage.js
 *
 * Wires the paper-companion homepage:
 *  - paper banner with SAE config
 *  - Table 1 row (method coverage by q<0.05)
 *  - 7-method coverage strip with click-to-learn dialogs
 *  - significance-aware AG Grid (21 method columns) with combo filter
 *  - Figure-2-style scatter coloured by significance tuple
 *  - UpSet-style bar of most common significance combinations
 */

const METHOD_DEFS = [
    {
        id: 1, name: "InterPro Protein", short: "IPR Prot", metric: "F1",
        summary: "Does a known InterPro family label predict which proteins the feature fires on?",
        detail: "At the protein level, we ask whether any single InterPro code is a significant predictor of which proteins carry the feature. We BH-correct the raw permutation p-values across all features.",
    },
    {
        id: 2, name: "InterPro Residue", short: "IPR Res", metric: "F1",
        summary: "Do InterPro domain fragments align with the residues the feature fires on?",
        detail: "At the residue level, we test whether the annotation's reported fragments coincide with the feature's activating residues. Uses BH q-values from the residue-level permutation null.",
    },
    {
        id: 3, name: "CATH Protein", short: "CATH Prot", metric: "F1",
        summary: "Does a CATH structural class explain which proteins the feature fires on?",
        detail: "CATH hierarchy labels (C / CA / CAT / CATH) are tested for protein-level predictivity; the highest-F1 level is reported. Raw p-values come from the permutation null; q is BH-corrected at viz startup.",
    },
    {
        id: 4, name: "CATH Residue", short: "CATH Res", metric: "F1",
        summary: "Do CATH domain boundaries align with the activating residues?",
        detail: "Residue-level test for CATH, analogous to InterPro residue. The highest-F1 CATH level is reported with its BH q-value.",
    },
    {
        id: 5, name: "Sequence Position", short: "Position", metric: "F1",
        summary: "Does the feature fire at a consistent position (e.g. N-terminus, interior 80%)?",
        detail: "We sweep 21 position predicates that encode only the residue's position along the sequence (not amino-acid identity). The best F1 is BH-corrected against a permutation null.",
    },
    {
        id: 6, name: "Sequence MEME Motif", short: "Motif", metric: "PR-AUC",
        summary: "Does the feature fire around a discoverable sequence motif?",
        detail: "MEME derives PWMs from windows around top-activating residues; we score each PWM via PR-AUC against the feature's activation label and take the best. Significance is the BH q-value on the PR-AUC against a permutation null.",
    },
    {
        id: 7, name: "Geometric", short: "Geom", metric: "PR-AUC",
        summary: "Can local Cα backbone geometry predict which residues the feature fires on?",
        detail: "A GBM is trained on a 44-dim local geometric feature vector (curvature, torsion, planarity, compactness, contacts, composition) and evaluated via PR-AUC on activation. This is the paper's novel annotation method.",
    },
];

const SCORE_FIELD = k => `m${k}_score`;
const LABEL_FIELD = k => `m${k}_label`;
const Q_FIELD     = k => `m${k}_q`;

const isSig = (q) => q != null && q < 0.05;

function fmt(val, decimals = 3) {
    if (val == null) return "—";
    return Number(val).toFixed(decimals);
}

function fmtQ(q) {
    if (q == null) return "—";
    if (q < 1e-3) return q.toExponential(1);
    return q.toFixed(3);
}

// ============================================================
// Paper banner + Table 1 row + method-coverage strip
// ============================================================

function renderPaperBanner(stats) {
    const grid = document.getElementById("sae-config-grid");
    if (!grid) return;
    const sae = stats.sae || {};
    const ds = stats.dataset || {};
    const parts = [
        ["Model", ds.esm_model || "—"],
        ["Layer", ds.esm_layer ?? "—"],
        ["Architecture", sae.architecture || "—"],
        ["Dict size", sae.dictionary_size != null ? sae.dictionary_size.toLocaleString() : (ds.num_features?.toLocaleString() ?? "—")],
        ["Expansion", sae.expansion_factor != null ? `${sae.expansion_factor}×` : "—"],
        ["Activation dim", sae.activation_dim ?? "—"],
        ["L1 penalty", sae.l1_penalty != null ? sae.l1_penalty.toFixed(4) : "—"],
        ["Learning rate", sae.lr != null ? sae.lr.toExponential(2) : "—"],
        ["Steps", sae.steps != null ? sae.steps.toLocaleString() : "—"],
        ["Run", sae.wandb_name || "—"],
        ["Proteins", ds.total_proteins?.toLocaleString() ?? "—"],
        ["Clusters", ds.total_clusters?.toLocaleString() ?? "—"],
    ];
    grid.innerHTML = parts.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

function renderLayerTable(coverage, stats) {
    const container = document.getElementById("layer-table-container");
    if (!container) return;
    const layer = stats.dataset?.esm_layer;
    const rows = [1, 2, 3, 4, 5, 6].map((L) => {
        const current = L === layer;
        const cells = current
            ? coverage.methods.map((m) => `<td><strong>${m.pct.toFixed(2)}%</strong></td>`).join("")
            : coverage.methods.map(() => `<td>—</td>`).join("");
        const total = current ? `<td><strong>${coverage.total_annotated_pct.toFixed(2)}%</strong></td>` : `<td>—</td>`;
        return `<tr class="${current ? "current-layer" : ""}"><td>Layer ${L}</td>${total}${cells}</tr>`;
    }).join("");
    const headers = `<th>Layer</th><th>Total</th>${coverage.methods.map((m) => `<th title="${m.name} (${m.metric})">${METHOD_DEFS[m.id - 1].short}</th>`).join("")}`;
    container.innerHTML = `<table class="layer-table"><thead><tr>${headers}</tr></thead><tbody>${rows}</tbody></table>`;
}

function renderMethodStrip(coverage) {
    const strip = document.getElementById("method-coverage-strip");
    if (!strip) return;
    strip.innerHTML = "";
    for (const m of coverage.methods) {
        const def = METHOD_DEFS[m.id - 1];
        const badge = document.createElement("span");
        badge.className = "method-badge" + (m.id === 7 ? " geometry-highlight" : "");
        badge.title = def.summary;
        badge.innerHTML = `<span class="name">${def.name}</span><span class="pct">${m.pct.toFixed(1)}%</span>`;
        badge.addEventListener("click", () => openMethodDialog(def, m));
        strip.appendChild(badge);
    }
    const tot = document.createElement("span");
    tot.className = "total-badge";
    tot.title = "Features where at least one method has q < 0.05";
    tot.innerHTML = `<span>Any method</span><span>${coverage.total_annotated_pct.toFixed(1)}%</span>`;
    strip.appendChild(tot);
}

function openMethodDialog(def, coverageEntry) {
    const dlg = document.getElementById("method-info-dialog");
    if (!dlg) return;
    document.getElementById("method-info-title").textContent = `${def.name}  ·  ${def.metric}`;
    document.getElementById("method-info-body").innerHTML = `
        <p>${def.detail}</p>
        <p><strong>${coverageEntry.n_significant.toLocaleString()} / ${coverageEntry.total.toLocaleString()}</strong>
        features have q&nbsp;&lt;&nbsp;0.05 for this method (${coverageEntry.pct.toFixed(2)}%).</p>
    `;
    if (typeof dlg.showModal === "function") {
        dlg.showModal();
    } else {
        dlg.setAttribute("open", "open");
    }
}

// ============================================================
// AG Grid
// ============================================================

function nullBottomComparator(a, b, _nA, _nB, desc) {
    if (a == null && b == null) return 0;
    if (a == null) return desc ? -1 : 1;
    if (b == null) return desc ? 1 : -1;
    if (typeof a === "string" || typeof b === "string") return String(a).localeCompare(String(b));
    return a - b;
}

function significanceCellClass(k) {
    return (params) => {
        const q = params.data?.[Q_FIELD(k)];
        if (q == null) return "null-cell";
        return q < 0.05 ? "sig-cell" : "not-sig-cell";
    };
}

function geometryHeaderClass(k) {
    return k === 7 ? "geometry-col" : undefined;
}

function scoreFormatter(params) {
    const v = params.value;
    if (v == null) return "—";
    return Number(v).toFixed(3);
}

function labelFormatter(params) {
    return params.value || "—";
}

function buildColumnDefs() {
    const cols = [
        {
            field: "feature_id", headerName: "ID", width: 80, pinned: "left",
            filter: "agNumberColumnFilter",
        },
        {
            field: "geometry_radar", headerName: "Profile", width: 66, pinned: "left",
            sortable: false, filter: false,
            cellRenderer: (params) => {
                const radar = params.value;
                if (!radar) return "";
                const div = document.createElement("div");
                div.style.cssText = "display:flex;align-items:center;justify-content:center;height:100%";
                const scores = [
                    radar.curvature || 0, radar.torsion || 0, radar.planarity || 0,
                    radar.compactness || 0, radar.contacts || 0, radar.composition || 0,
                ];
                renderRadarGlyph(div, scores, { size: 48, showLabels: false });
                return div;
            },
        },
        {
            field: "pct_proteins_activated", headerName: "% Prot.", width: 90,
            valueFormatter: (p) => p.value == null ? "—" : Number(p.value).toFixed(1),
            comparator: nullBottomComparator, filter: "agNumberColumnFilter",
        },
        {
            field: "max_activation", headerName: "Max act.", width: 95,
            valueFormatter: (p) => p.value == null ? "—" : Number(p.value).toFixed(3),
            comparator: nullBottomComparator, filter: "agNumberColumnFilter",
        },
    ];
    for (const def of METHOD_DEFS) {
        const k = def.id;
        cols.push({
            field: SCORE_FIELD(k),
            headerName: `${def.short}`,
            headerTooltip: `${def.name} — ${def.metric}`,
            width: 95,
            valueFormatter: scoreFormatter,
            cellClass: significanceCellClass(k),
            headerClass: geometryHeaderClass(k),
            comparator: nullBottomComparator,
            filter: "agNumberColumnFilter",
        });
        cols.push({
            field: LABEL_FIELD(k),
            headerName: `${def.short} label`,
            headerTooltip: `Top label under ${def.name}`,
            width: 170,
            valueFormatter: labelFormatter,
            cellClass: significanceCellClass(k),
            headerClass: geometryHeaderClass(k),
            filter: "agTextColumnFilter",
        });
    }
    return cols;
}

let _gridApi = null;
let _currentSigFilter = "all";
let _currentHideZeroSig = false;

function rowSignificanceFlags(row) {
    const sigKs = [];
    for (let k = 1; k <= 7; k++) {
        if (isSig(row[Q_FIELD(k)])) sigKs.push(k);
    }
    const bioSig = sigKs.some((k) => k <= 6);
    const geomSig = sigKs.includes(7);
    return { sigKs, bioSig, geomSig, nSig: sigKs.length };
}

function matchesSigFilter(row) {
    const { bioSig, geomSig, nSig } = rowSignificanceFlags(row);
    if (_currentHideZeroSig && nSig === 0) return false;
    switch (_currentSigFilter) {
        case "geom_only": return geomSig && !bioSig;
        case "bio_only":  return bioSig && !geomSig;
        case "both":      return geomSig && bioSig;
        case "none":      return nSig === 0;
        default:          return true;
    }
}

function refreshRowCount() {
    if (!_gridApi) return;
    const total = _gridApi.getDisplayedRowCount();
    const el = document.getElementById("row-count");
    if (el) el.textContent = `${total.toLocaleString()} row${total === 1 ? "" : "s"} shown`;
}

function initGrid(rowData) {
    const gridDiv = document.getElementById("feature-grid");
    const gridOptions = {
        columnDefs: buildColumnDefs(),
        rowData: rowData,
        defaultColDef: { sortable: true, resizable: true },
        rowHeight: 56,
        animateRows: false,
        suppressCellFocus: true,
        isExternalFilterPresent: () => _currentSigFilter !== "all" || _currentHideZeroSig,
        doesExternalFilterPass: (node) => matchesSigFilter(node.data),
        onRowClicked: (event) => {
            const fid = event.data.feature_id;
            window.location.href = `/feature/${fid}`;
        },
        onGridReady: (event) => {
            _gridApi = event.api;
            refreshRowCount();
        },
        onFilterChanged: refreshRowCount,
        onSortChanged: refreshRowCount,
        onModelUpdated: refreshRowCount,
    };
    agGrid.createGrid(gridDiv, gridOptions);
}

// ============================================================
// Figure-2 style significance scatter
// ============================================================

function renderFig2Scatter(index) {
    const div = document.getElementById("fig2-scatter");
    if (!div) return;
    const classify = (row) => {
        const { bioSig, geomSig } = rowSignificanceFlags(row);
        if (geomSig && !bioSig) return "geom_only";
        if (bioSig && !geomSig) return "bio_only";
        if (bioSig && geomSig)  return "both";
        return "none";
    };
    const tint = {
        geom_only: "#f59e0b",
        bio_only:  "#2563eb",
        both:      "#7c3aed",
        none:      "#9ca3af",
    };
    const groupLabels = {
        geom_only: "Geometry only",
        bio_only:  "Bio only",
        both:      "Both",
        none:      "Neither",
    };
    const groups = { geom_only: [], bio_only: [], both: [], none: [] };
    for (const row of index) {
        if (row.m7_score == null) continue;
        const bestBio = [row.m1_score, row.m2_score, row.m3_score, row.m4_score, row.m5_score, row.m6_score]
            .filter((v) => v != null);
        if (bestBio.length === 0) continue;
        groups[classify(row)].push({
            x: row.m7_score, y: Math.max(...bestBio), fid: row.feature_id,
        });
    }
    const traces = Object.entries(groups)
        .filter(([, rows]) => rows.length > 0)
        .map(([k, rows]) => ({
            x: rows.map(r => r.x),
            y: rows.map(r => r.y),
            customdata: rows.map(r => r.fid),
            text: rows.map(r => `Feature ${r.fid}`),
            mode: "markers",
            type: "scatter",
            name: `${groupLabels[k]} (${rows.length.toLocaleString()})`,
            marker: {
                size: k === "none" ? 3 : 5,
                color: tint[k],
                opacity: k === "none" ? 0.3 : 0.75,
                line: { width: 0 },
            },
            hovertemplate: "%{text}<br>Geom PR-AUC: %{x:.3f}<br>Best bio F1: %{y:.3f}<extra></extra>",
        }));
    Plotly.newPlot(div, traces, {
        title: { text: "Figure 2 — Geometry vs best biological annotation", font: { size: 13 } },
        xaxis: { title: "Geometry PR-AUC (method 7)", gridcolor: "#e9ecef", zeroline: false },
        yaxis: { title: "max F1 across methods 1–6", gridcolor: "#e9ecef", zeroline: false },
        margin: { t: 45, r: 20, b: 50, l: 55 },
        hovermode: "closest",
        paper_bgcolor: "#fff",
        plot_bgcolor: "#f8f9fa",
        legend: { x: 0.02, y: 0.98, bgcolor: "rgba(255,255,255,0.85)", font: { size: 10 } },
        font: { family: "sans-serif" },
    }, { responsive: true, displayModeBar: false });

    div.on("plotly_click", (data) => {
        const fid = data.points[0]?.customdata;
        if (fid != null) window.location.href = `/feature/${fid}`;
    });
}

// ============================================================
// UpSet-style combination bar
// ============================================================

function renderUpsetBar(index) {
    const div = document.getElementById("upset-chart");
    if (!div) return;
    const counts = new Map();
    for (const row of index) {
        const key = rowSignificanceFlags(row).sigKs.join("·") || "none";
        counts.set(key, (counts.get(key) || 0) + 1);
    }
    const entries = [...counts.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 20)
        .reverse();
    const labels = entries.map(([key]) => key === "none" ? "(none)" : key);
    const values = entries.map(([, n]) => n);
    const hovertext = entries.map(([key, n]) => {
        if (key === "none") return `No significant methods · ${n.toLocaleString()} features`;
        const names = key.split("·").map((k) => METHOD_DEFS[parseInt(k, 10) - 1].short).join(" + ");
        return `${names}<br>${n.toLocaleString()} features`;
    });
    Plotly.newPlot(div, [{
        y: labels,
        x: values,
        type: "bar",
        orientation: "h",
        customdata: entries.map(([key]) => key),
        text: hovertext,
        hovertemplate: "%{text}<extra></extra>",
        marker: {
            color: entries.map(([key]) => {
                if (key === "none") return "#9ca3af";
                const ks = key.split("·").map(Number);
                const geom = ks.includes(7);
                const bio = ks.some((k) => k <= 6);
                if (geom && bio)  return "#7c3aed";
                if (geom)         return "#f59e0b";
                return "#2563eb";
            }),
        },
    }], {
        title: { text: "Top 20 significance combinations", font: { size: 13 } },
        xaxis: { title: "Features", gridcolor: "#e9ecef" },
        yaxis: { tickfont: { size: 10 } },
        margin: { t: 45, r: 20, b: 50, l: 100 },
        paper_bgcolor: "#fff",
        plot_bgcolor: "#f8f9fa",
        font: { family: "sans-serif" },
    }, { responsive: true, displayModeBar: false });

    div.on("plotly_click", (data) => {
        const key = data.points[0]?.customdata;
        if (key == null) return;
        const sel = document.getElementById("sig-filter");
        if (key === "none") {
            sel.value = "none";
        } else {
            const ks = key.split("·").map(Number);
            const geom = ks.includes(7);
            const bio = ks.some((k) => k <= 6);
            sel.value = geom && bio ? "both" : geom ? "geom_only" : "bio_only";
        }
        sel.dispatchEvent(new Event("change"));
        document.getElementById("table-section").scrollIntoView({ behavior: "smooth" });
    });
}

// ============================================================
// Main
// ============================================================

document.addEventListener("DOMContentLoaded", async () => {
    try {
        const [statsRes, idxRes, covRes] = await Promise.all([
            fetch("/api/stats"),
            fetch("/api/index"),
            fetch("/api/method-coverage"),
        ]);
        if (!statsRes.ok) throw new Error(`/api/stats: ${statsRes.status}`);
        if (!idxRes.ok)   throw new Error(`/api/index: ${idxRes.status}`);
        if (!covRes.ok)   throw new Error(`/api/method-coverage: ${covRes.status}`);

        const stats = await statsRes.json();
        const index = await idxRes.json();
        const coverage = await covRes.json();

        const ds = stats.dataset || {};
        document.getElementById("subtitle").textContent =
            `${ds.esm_model || "ESM"} layer ${ds.esm_layer ?? "?"} · ` +
            `${index.length.toLocaleString()} features · ` +
            `${coverage.total_annotated_n.toLocaleString()} (${coverage.total_annotated_pct.toFixed(1)}%) annotated by ≥1 method`;

        renderPaperBanner(stats);
        renderLayerTable(coverage, stats);
        renderMethodStrip(coverage);
        initGrid(index);
        renderFig2Scatter(index);
        renderUpsetBar(index);

        // Filter controls
        document.getElementById("sig-filter").addEventListener("change", (e) => {
            _currentSigFilter = e.target.value;
            if (_gridApi) _gridApi.onFilterChanged();
        });
        document.getElementById("show-only-significant").addEventListener("change", (e) => {
            _currentHideZeroSig = e.target.checked;
            if (_gridApi) _gridApi.onFilterChanged();
        });
    } catch (err) {
        console.error("homepage load failed:", err);
        const sub = document.getElementById("subtitle");
        if (sub) sub.textContent = `Error: ${err.message}`;
    }
});
