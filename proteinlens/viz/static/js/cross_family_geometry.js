/**
 * cross_family_geometry.js
 *
 * Renders "Geometry annotates features with missing DB annotations".
 * Classification is computed client-side from /api/index so the page
 * reflects the current q-values rather than a precomputed snapshot.
 *
 *   sig(k)               = row[`m${k}_q`] != null && row[`m${k}_q`] < 0.05
 *   db_silent            = none of m1..m4 are significant
 *   geom_sig             = m7 is significant
 *   DB-silent geom-annot = db_silent && geom_sig
 *   purely geometric     = DB-silent geom-annot && !sig(5) && !sig(6)
 */

const STRUCT_COLORS = {
    Curvature: "#2563eb",
    Torsion: "#dc2626",
    Planarity: "#16a34a",
    Compactness: "#f59e0b",
    Contacts: "#7c3aed",
    Composition: "#0ea5e9",
    Other: "#9ca3af",
};

const sig = (row, k) => row[`m${k}_q`] != null && row[`m${k}_q`] < 0.05;
const dbSilent = (r) => !sig(r, 1) && !sig(r, 2) && !sig(r, 3) && !sig(r, 4);
const geomSig  = (r) => sig(r, 7);
const isDbSilentGeomAnnotatable = (r) => dbSilent(r) && geomSig(r);
const isPurelyGeometric = (r) => isDbSilentGeomAnnotatable(r) && !sig(r, 5) && !sig(r, 6);

function fmt(v, d = 3) { return v == null ? "—" : Number(v).toFixed(d); }
function fmtQ(q) {
    if (q == null) return "—";
    if (q < 1e-3) return q.toExponential(1);
    return q.toFixed(3);
}

function maybeRedirect() {
    if (window.location.pathname === "/cross-family-geometry") {
        history.replaceState({}, "", "/geometry-fills-missing-db");
    }
}

function categoryHead(row) {
    const c = row.structural_category || row.m7_label || "Other";
    const head = c.split(/[\s(]/)[0];
    return STRUCT_COLORS[head] ? head : "Other";
}

function renderGlobalStats(total, dbSilentGeomN, purelyGeomN) {
    const c = document.getElementById("global-stats");
    c.innerHTML = `
        <article class="stat-callout">
            <div class="label">Total features</div>
            <div class="value">${total.toLocaleString()}</div>
            <div class="sub">In this SAE run</div>
        </article>
        <article class="stat-callout flag-green">
            <div class="label">DB-silent geometry-annotatable</div>
            <div class="value">${dbSilentGeomN.toLocaleString()}</div>
            <div class="sub">${(100 * dbSilentGeomN / total).toFixed(2)}% of features</div>
        </article>
        <article class="stat-callout flag-green">
            <div class="label">Purely geometric</div>
            <div class="value">${purelyGeomN.toLocaleString()}</div>
            <div class="sub">Position + MEME also not significant</div>
        </article>
    `;
}

function renderStructCatChart(index) {
    const allGeom = index.filter(geomSig);
    const dbSilentGeom = index.filter(isDbSilentGeomAnnotatable);
    const buckets = Object.keys(STRUCT_COLORS);
    const countsAll = buckets.map(b => allGeom.filter(r => categoryHead(r) === b).length);
    const countsDbSilent = buckets.map(b => dbSilentGeom.filter(r => categoryHead(r) === b).length);
    Plotly.newPlot("struct-cat-chart", [
        {
            x: buckets, y: countsAll, type: "bar",
            name: `All geometry-significant (${allGeom.length.toLocaleString()})`,
            marker: { color: "#6366f1" },
        },
        {
            x: buckets, y: countsDbSilent, type: "bar",
            name: `DB-silent geometry-annotatable (${dbSilentGeom.length.toLocaleString()})`,
            marker: { color: "#10b981" },
        },
    ], {
        barmode: "group",
        xaxis: { title: "Top structural category" },
        yaxis: { title: "Features" },
        legend: { x: 0.02, y: 0.98, font: { size: 11 } },
        paper_bgcolor: "#fff",
        plot_bgcolor: "#f8f9fa",
        margin: { t: 20, r: 20, b: 60, l: 60 },
        font: { family: "sans-serif" },
    }, { responsive: true, displayModeBar: false });
}

function renderResidueF1Hist(index) {
    const bestResF1 = (row) => {
        const vals = [row.m2_score, row.m4_score, row.m5_score, row.m6_score].filter(v => v != null);
        return vals.length ? Math.max(...vals) : null;
    };
    const allGeom = index.filter(geomSig).map(bestResF1).filter(v => v != null);
    const silent = index.filter(isDbSilentGeomAnnotatable).map(bestResF1).filter(v => v != null);
    const pure = index.filter(isPurelyGeometric).map(bestResF1).filter(v => v != null);
    Plotly.newPlot("residue-f1-chart", [
        { x: allGeom, type: "histogram", opacity: 0.55, name: `All geom-sig (${allGeom.length.toLocaleString()})`, marker: { color: "#6366f1" }, xbins: { size: 0.05 } },
        { x: silent,  type: "histogram", opacity: 0.75, name: `DB-silent geom (${silent.length.toLocaleString()})`, marker: { color: "#10b981" }, xbins: { size: 0.05 } },
        { x: pure,    type: "histogram", opacity: 0.9,  name: `Purely geom (${pure.length.toLocaleString()})`, marker: { color: "#f59e0b" }, xbins: { size: 0.05 } },
    ], {
        barmode: "overlay",
        xaxis: { title: "max residue-level bio F1", range: [0, 1] },
        yaxis: { title: "Features" },
        paper_bgcolor: "#fff",
        plot_bgcolor: "#f8f9fa",
        margin: { t: 20, r: 20, b: 60, l: 60 },
        legend: { x: 0.55, y: 0.98, font: { size: 11 } },
        font: { family: "sans-serif" },
    }, { responsive: true, displayModeBar: false });
}

const METHOD_SHORT = ["IPR Prot", "IPR Res", "CATH Prot", "CATH Res", "Pos", "Motif", "Geom"];

function renderCards(features, limit = 24) {
    const container = document.getElementById("cross-family-container");
    if (features.length === 0) {
        container.innerHTML = '<p class="secondary">No features match the DB-silent geometry-annotatable criterion.</p>';
        return;
    }
    container.innerHTML = "";
    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:.75rem;";
    for (const row of features.slice(0, limit)) {
        const card = document.createElement("a");
        card.href = `/feature/${row.feature_id}`;
        card.className = "case-study-tile";
        if (isPurelyGeometric(row)) {
            card.style.borderColor = "#f59e0b";
            card.style.borderWidth = "2px";
        }
        const radar = document.createElement("div");
        radar.style.cssText = "display:flex;justify-content:center;margin-bottom:.25rem;";
        if (row.geometry_radar && typeof renderRadarGlyph === "function") {
            const s = row.geometry_radar;
            renderRadarGlyph(radar, [
                s.curvature || 0, s.torsion || 0, s.planarity || 0,
                s.compactness || 0, s.contacts || 0, s.composition || 0,
            ], { size: 96, showLabels: false });
        }
        const chips = [1, 2, 3, 4, 5, 6, 7].map(k => {
            const q = row[`m${k}_q`];
            const cls = q == null ? "null" : q < 0.05 ? "sig" : "";
            return `<span class="q-chip ${cls}" title="${METHOD_SHORT[k - 1]}">${k}: ${fmtQ(q)}</span>`;
        }).join(" ");
        const title = `Feature ${row.feature_id}` + (isPurelyGeometric(row)
            ? ' <span style="color:#f59e0b">· purely geom</span>' : "");
        card.appendChild(radar);
        const inner = document.createElement("div");
        inner.innerHTML = `
            <h4>${title}</h4>
            <p>${row.structural_category || "—"}</p>
            <div style="display:flex;flex-wrap:wrap;gap:.2rem;margin-top:.4rem;">${chips}</div>
        `;
        card.appendChild(inner);
        grid.appendChild(card);
    }
    container.appendChild(grid);
}

function renderFullTable(features) {
    const container = document.getElementById("all-features-table");
    if (features.length === 0) {
        container.innerHTML = '<p class="secondary">No features match.</p>';
        return;
    }
    const rows = features.map((r) => {
        const cells = [1, 2, 3, 4, 5, 6, 7].map((k) => {
            const q = r[`m${k}_q`];
            const score = r[`m${k}_score`];
            const style = q == null ? ""
                : q < 0.05 ? 'style="background:#d1fae5;color:#065f46;font-weight:600"'
                : 'style="color:#9ca3af"';
            return `<td ${style}>${fmt(score, 3)}<br><span style="font-size:.7rem">q=${fmtQ(q)}</span></td>`;
        }).join("");
        const pure = isPurelyGeometric(r) ? "✓" : "";
        return `<tr onclick="location.href='/feature/${r.feature_id}'" style="cursor:pointer">
            <td>${r.feature_id}</td>
            <td>${r.structural_category || "—"}</td>
            <td style="text-align:center;color:#f59e0b;font-weight:700">${pure}</td>
            ${cells}
        </tr>`;
    }).join("");
    container.innerHTML = `
        <table style="width:100%;border-collapse:collapse;font-size:.82rem;">
            <thead style="background:#f9fafb">
                <tr>
                    <th>ID</th><th>Top structural category</th><th>Purely geom</th>
                    <th>IPR Prot</th><th>IPR Res</th><th>CATH Prot</th><th>CATH Res</th>
                    <th>Pos</th><th>Motif</th><th>Geom</th>
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}

document.addEventListener("DOMContentLoaded", async () => {
    maybeRedirect();
    try {
        const indexRes = await fetch("/api/index");
        if (!indexRes.ok) throw new Error(`/api/index: ${indexRes.status}`);
        const index = await indexRes.json();

        const silent = index.filter(isDbSilentGeomAnnotatable);
        const pure = silent.filter(isPurelyGeometric);

        renderGlobalStats(index.length, silent.length, pure.length);
        renderStructCatChart(index);
        renderResidueF1Hist(index);

        silent.sort((a, b) => (b.m7_score || 0) - (a.m7_score || 0));
        renderCards(silent);
        renderFullTable(silent);
    } catch (err) {
        console.error("cross-family-geometry load failed:", err);
        document.getElementById("global-stats").innerHTML =
            `<p style="color:red">Error: ${err.message}</p>`;
    }
});
