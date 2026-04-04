/**
 * case_studies.js — Renders the case study family list page.
 *
 * Fetches /api/case-study-families and renders a card for each family.
 * Clicking a card navigates to /case-studies/{annotation_code}.
 */

function fmtVal(v, decimals = 3) {
    if (v === null || v === undefined) return "\u2014";
    return Number(v).toFixed(decimals);
}

function renderFamilyCard(family) {
    const card = document.createElement("article");
    card.className = "info-card";
    card.style.cursor = "pointer";
    card.style.transition = "box-shadow 0.2s";
    card.addEventListener("mouseenter", () => { card.style.boxShadow = "0 2px 12px rgba(0,0,0,0.15)"; });
    card.addEventListener("mouseleave", () => { card.style.boxShadow = ""; });
    card.addEventListener("click", () => {
        window.location.href = `/case-studies/${family.annotation_code}`;
    });

    const diverseBadge = family.geom_diverse
        ? '<span class="badge badge-done">Geometry-diverse</span>'
        : '<span class="badge badge-pending">Same top feature</span>';

    const memberRows = family.members.map(m =>
        `<tr>
            <td><a href="/feature/${m.feature_id}" onclick="event.stopPropagation()">${m.feature_id}</a></td>
            <td>${fmtVal(m.pct_proteins_activated, 1)}%</td>
            <td>${fmtVal(m.interpro_res_f1)}</td>
            <td>${fmtVal(m.geom_pr_auc)}</td>
            <td style="font-family:monospace;font-size:0.8rem">${m.top_geometric_feature}</td>
        </tr>`
    ).join("");

    card.innerHTML = `
        <header>
            <strong>${family.annotation_name}</strong>
            <span class="secondary" style="margin-left:0.5rem;font-size:0.8rem">${family.annotation_code}</span>
        </header>
        <div style="margin:0.5rem 0;">
            ${diverseBadge}
            <span class="badge badge-count">${family.n_nodes} nodes</span>
            <span class="badge badge-count">${family.n_unique_top_geom} unique geom features</span>
            <span class="badge badge-count">cos sim: ${fmtVal(family.mean_cosine_similarity, 2)}</span>
        </div>
        <table role="grid" style="font-size:0.8rem;margin:0;">
            <thead>
                <tr>
                    <th>Node</th>
                    <th>Coverage</th>
                    <th>InterPro Res F1</th>
                    <th>Geom PR-AUC</th>
                    <th>Top Geom Feature</th>
                </tr>
            </thead>
            <tbody>${memberRows}</tbody>
        </table>
    `;

    return card;
}

document.addEventListener("DOMContentLoaded", async () => {
    try {
        const res = await fetch("/api/case-study-families");
        if (!res.ok) throw new Error(`Failed: ${res.status}`);
        const data = await res.json();

        // Summary stats with methodology
        const nt = data.null_thresholds;
        document.getElementById("summary-stats").innerHTML =
            `<strong>${data.n_qualifying_features} sparse features</strong> (&le;${fmtVal(nt.max_pct_activated, 0)}% coverage) ` +
            `with InterPro residue F1 &gt; ${fmtVal(nt.interpro_res_f1, 4)} (null p95) ` +
            `and geometry PR-AUC &gt; ${fmtVal(nt.geom_pr_auc, 4)} ` +
            `&mdash; grouped into <strong>${data.n_families} families</strong> with 2+ nodes sharing the same InterPro annotation.` +
            `<br><span style="font-size:0.8rem;">Null thresholds estimated from features with &lt;1% protein activation (95th percentile). ` +
            `These features have both strong InterPro and geometry signals; within each family, nodes capture <em>different geometric properties</em> of the same domain.</span>`;

        // Render family cards
        const container = document.getElementById("families-container");
        container.innerHTML = "";
        const grid = document.createElement("div");
        grid.style.display = "grid";
        grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(500px, 1fr))";
        grid.style.gap = "1rem";

        for (const family of data.families) {
            grid.appendChild(renderFamilyCard(family));
        }
        container.appendChild(grid);
    } catch (err) {
        console.error(err);
        document.getElementById("families-container").innerHTML =
            `<p style="color:red">Error: ${err.message}. Run <code>python scripts/build_case_studies.py --data-dir &lt;data_dir&gt;</code> first.</p>`;
    }
});
