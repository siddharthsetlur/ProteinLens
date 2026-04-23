/**
 * meme_case_studies.js — Renders the MEME case study family list.
 *
 * Fetches /api/meme-case-study-families and renders a card per family.
 * Clicking a card navigates to /meme-case-studies/{representative_consensus}.
 */

function fmtVal(v, decimals = 3) {
    if (v === null || v === undefined) return "\u2014";
    return Number(v).toFixed(decimals);
}

/** True when >60% of consensus chars are N (degenerate, no real motif). */
function isDegenerateConsensus(consensus) {
    if (!consensus) return true;
    const nCount = (consensus.match(/N/g) || []).length;
    return nCount / consensus.length > 0.6;
}

function renderFamilyCard(family) {
    const card = document.createElement("article");
    card.className = "info-card";
    card.style.cursor = "pointer";
    card.style.transition = "box-shadow 0.2s";
    card.addEventListener("mouseenter", () => { card.style.boxShadow = "0 2px 12px rgba(0,0,0,0.15)"; });
    card.addEventListener("mouseleave", () => { card.style.boxShadow = ""; });
    card.addEventListener("click", () => {
        window.location.href = `/meme-case-studies/${encodeURIComponent(family.representative_consensus)}`;
    });

    const diverseBadge = family.geom_diverse
        ? '<span class="badge badge-done">Geometry-diverse (cos &lt; 0.5)</span>'
        : '<span class="badge badge-pending">Not geometry-diverse (cos &ge; 0.5)</span>';

    const memberRows = family.members.map(m =>
        `<tr>
            <td><a href="/feature/${m.feature_id}" onclick="event.stopPropagation()">${m.feature_id}</a></td>
            <td>${fmtVal(m.pct_proteins_activated, 1)}%</td>
            <td style="font-family:monospace;font-size:0.8rem">${m.consensus}</td>
            <td>${fmtVal(m.motif_pr_auc)}</td>
            <td>${fmtVal(m.geom_pr_auc)}</td>
            <td style="font-size:0.75rem;color:#6b7280" title="${m.top_geometric_feature}">${(m.top_geometric_feature || "—").replace(/_/g, " ")}</td>
        </tr>`
    ).join("");

    card.innerHTML = `
        <header>
            <strong style="font-family:monospace">${family.representative_consensus}</strong>
            <span class="secondary" style="margin-left:0.5rem;font-size:0.8rem">representative MEME consensus</span>
        </header>
        <div style="margin:0.5rem 0;">
            ${diverseBadge}
            <span class="badge badge-count">${family.n_nodes} nodes</span>
            <span class="badge badge-count">${family.n_unique_top_geom} unique geom features</span>
            <span class="badge badge-count">cos sim: ${fmtVal(family.mean_cosine_similarity, 2)}</span>
        </div>
        <div style="overflow-x:auto">
        <table role="grid" style="font-size:0.8rem;margin:0;min-width:420px">
            <thead>
                <tr>
                    <th>Node</th>
                    <th>Coverage</th>
                    <th>Consensus</th>
                    <th>MEME PR-AUC</th>
                    <th>Geom PR-AUC</th>
                    <th>Top geom feature</th>
                </tr>
            </thead>
            <tbody>${memberRows}</tbody>
        </table>
        </div>
    `;

    return card;
}

document.addEventListener("DOMContentLoaded", async () => {
    try {
        const res = await fetch("/api/meme-case-study-families");
        if (!res.ok) throw new Error(`Failed: ${res.status}`);
        const data = await res.json();

        const qg = data.q_gate || {};
        const maxPct = qg.max_pct_activated != null ? qg.max_pct_activated : 20;
        const grouping = data.grouping || {};
        document.getElementById("summary-stats").innerHTML =
            `<strong>${data.n_qualifying_features} sparse features</strong> (&le;${fmtVal(maxPct, 0)}% coverage) ` +
            `with MEME-motif q&nbsp;&lt;&nbsp;0.05 AND geometry q&nbsp;&lt;&nbsp;0.05 ` +
            `&mdash; grouped into <strong>${data.n_families} families</strong> ` +
            `(<strong>${data.n_geom_diverse_families} geometry-diverse</strong>) via ` +
            `<em>${grouping.method || "consensus similarity"}</em>` +
            (grouping.max_edit_distance !== undefined ? ` (edit distance &le; ${grouping.max_edit_distance})` : "") + `.` +
            `<br><span style="font-size:0.8rem;">Both annotation methods are significant (BH-corrected q&nbsp;&lt;&nbsp;0.05). ` +
            `In diverse families, nodes share the same MEME sequence motif but capture <em>different geometric properties</em>.</span>`;

        const container = document.getElementById("families-container");
        container.innerHTML = "";
        const grid = document.createElement("div");
        grid.style.display = "grid";
        grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(480px, 1fr))";
        grid.style.gap = "1rem";

        const shown = data.families.filter(f => !isDegenerateConsensus(f.representative_consensus));
        const nFiltered = data.families.length - shown.length;
        if (nFiltered > 0) {
            const note = document.createElement("p");
            note.className = "secondary";
            note.style.fontSize = "0.8rem";
            note.textContent = `${nFiltered} family/families with degenerate (all-N) consensus hidden.`;
            container.appendChild(note);
        }
        for (const family of shown) {
            grid.appendChild(renderFamilyCard(family));
        }
        container.appendChild(grid);
    } catch (err) {
        console.error(err);
        document.getElementById("families-container").innerHTML =
            `<p style="color:red">Error: ${err.message}. Run <code>python scripts/build_meme_case_studies.py --data-dir &lt;data_dir&gt;</code> first.</p>`;
    }
});
