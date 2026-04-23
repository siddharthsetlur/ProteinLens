/**
 * radar_glyph.js — Compact radar/spider glyph for geometry feature profiles.
 *
 * Aggregates 44 residue-level geometry feature importances into 6 semantic
 * categories, then renders a small SVG radar chart. The shape provides instant
 * visual recognition of a feature's geometric "personality."
 *
 * Categories:
 *   Curvature  (blue)    — backbone bending
 *   Torsion    (red)     — backbone twist
 *   Planarity  (green)   — local flatness
 *   Compactness(orange)  — end-to-end / tangent alignment
 *   Contacts   (purple)  — spatial proximity / packing
 *   Composition(teal)    — amino acid makeup
 */

// ============================================================
// Category definitions: map raw feature names → 6 categories
// ============================================================

const RADAR_CATEGORIES = [
    {
        name: "Curvature",
        color: "#3b82f6",
        features: [
            "curvature_mean", "curvature_max", "curvature_std",
            "curv_N_third", "curv_centre_third", "curv_C_third",
            "narrow_curvature_mean", "narrow_curvature_max",
            "wide_curvature_mean", "wide_curvature_max",
        ],
    },
    {
        name: "Torsion",
        color: "#ef4444",
        features: [
            "torsion_mean", "torsion_std", "torsion_frac_pos",
            "tors_N_third", "tors_centre_third", "tors_C_third",
            "narrow_torsion_mean", "narrow_torsion_std",
            "wide_torsion_mean", "wide_torsion_std",
        ],
    },
    {
        name: "Planarity",
        color: "#22c55e",
        features: [
            "planarity_mean", "planarity_std",
            "plan_N_third", "plan_centre_third", "plan_C_third",
        ],
    },
    {
        name: "Compactness",
        color: "#f97316",
        features: [
            "tangent_alignment", "end_to_end_ratio",
            "narrow_tangent_alignment", "narrow_end_to_end_ratio",
            "wide_tangent_alignment", "wide_end_to_end_ratio",
        ],
    },
    {
        name: "Contacts",
        color: "#8b5cf6",
        features: [
            "contact_density_8A", "contact_density_12A",
            "long_range_contacts_8A", "long_range_contacts_12A",
            "max_seq_sep_contact_8A", "mean_seq_sep_contact_8A",
            "contact_order_local", "min_spatial_dist_long",
        ],
    },
    {
        name: "Composition",
        color: "#14b8a6",
        features: [
            "frac_hydrophobic", "frac_charged", "frac_polar",
            "frac_gly_pro", "frac_aromatic",
        ],
    },
];

// ============================================================
// Aggregation: raw feature importances → 6D category vector
// ============================================================

/**
 * Aggregate raw feature importances into category scores.
 *
 * For each category, sums the importances of all member features, then
 * normalizes so the 6 values sum to 1 (relative profile).
 *
 * @param {Object} importances - Dict mapping feature names → importance values.
 * @returns {number[]} Array of 6 category scores (sum to 1), or null if empty.
 */
function aggregateToCategories(importances) {
    if (!importances || Object.keys(importances).length === 0) return null;

    const scores = RADAR_CATEGORIES.map((cat) => {
        let sum = 0;
        for (const feat of cat.features) {
            sum += importances[feat] || 0;
        }
        return sum;
    });

    const total = scores.reduce((a, b) => a + b, 0);
    if (total === 0) return null;

    return scores.map((s) => s / total);
}

// ============================================================
// SVG Radar Glyph Renderer
// ============================================================

/**
 * Render a radar/spider glyph as an inline SVG element.
 *
 * @param {HTMLElement} container - DOM element to append the SVG to.
 * @param {number[]} categoryScores - 6 normalized scores (0-1 each, sum to 1).
 * @param {Object} [opts] - Options.
 * @param {number} [opts.size=60] - Width/height of the SVG in px.
 * @param {boolean} [opts.showLabels=false] - Show category name labels.
 * @param {boolean} [opts.showTooltip=true] - Add a title tooltip.
 * @param {string} [opts.className=""] - Extra CSS class for the SVG element.
 * @returns {SVGElement} The created SVG element.
 */
function renderRadarGlyph(container, categoryScores, opts = {}) {
    const size = opts.size || 60;
    const showLabels = opts.showLabels || false;
    const showTooltip = opts.showTooltip !== false;
    const className = opts.className || "";

    const n = RADAR_CATEGORIES.length; // 6
    // For labeled glyphs, use a larger viewBox so labels sit outside the chart area
    const labelPad = showLabels ? 50 : 0;
    const vbSize = size + labelPad * 2;
    const cx = vbSize / 2;
    const cy = vbSize / 2;
    const margin = size * 0.1;
    const maxR = (size / 2) - margin;

    // Scale: max category score fills ~85% of the radius.
    // This keeps the polygon large and visible while leaving headroom.
    const maxVal = Math.max(...categoryScores);
    const scale = maxVal > 0 ? (maxR * 0.85) / maxVal : maxR;

    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("width", vbSize);
    svg.setAttribute("height", vbSize);
    svg.setAttribute("viewBox", `0 0 ${vbSize} ${vbSize}`);
    if (className) svg.setAttribute("class", className);
    svg.style.display = "inline-block";
    svg.style.verticalAlign = "middle";

    // Tooltip (category names only)
    if (showTooltip) {
        const sorted = RADAR_CATEGORIES.map((cat, i) => ({ name: cat.name, score: categoryScores[i] }))
            .filter((c) => c.score > 0.05)
            .sort((a, b) => b.score - a.score);
        const titleEl = document.createElementNS(ns, "title");
        titleEl.textContent = sorted.map((c) => c.name).join(", ");
        svg.appendChild(titleEl);
    }

    // Helper: angle for axis i (start from top, clockwise)
    const angle = (i) => (Math.PI * 2 * i) / n - Math.PI / 2;
    const px = (i, r) => cx + Math.cos(angle(i)) * r;
    const py = (i, r) => cy + Math.sin(angle(i)) * r;

    // Background grid rings (2 concentric circles)
    for (const frac of [0.5, 1.0]) {
        const ring = document.createElementNS(ns, "circle");
        ring.setAttribute("cx", cx);
        ring.setAttribute("cy", cy);
        ring.setAttribute("r", maxR * frac);
        ring.setAttribute("fill", "none");
        ring.setAttribute("stroke", "#e5e7eb");
        ring.setAttribute("stroke-width", "0.5");
        svg.appendChild(ring);
    }

    // Axis lines
    for (let i = 0; i < n; i++) {
        const line = document.createElementNS(ns, "line");
        line.setAttribute("x1", cx);
        line.setAttribute("y1", cy);
        line.setAttribute("x2", px(i, maxR));
        line.setAttribute("y2", py(i, maxR));
        line.setAttribute("stroke", "#d1d5db");
        line.setAttribute("stroke-width", "0.5");
        svg.appendChild(line);
    }

    // Data polygon (filled)
    const points = categoryScores.map((val, i) => {
        const r = val * scale;
        return `${px(i, r)},${py(i, r)}`;
    }).join(" ");

    const polygon = document.createElementNS(ns, "polygon");
    polygon.setAttribute("points", points);
    polygon.setAttribute("fill", "rgba(99, 102, 241, 0.25)");
    polygon.setAttribute("stroke", "rgba(99, 102, 241, 0.8)");
    polygon.setAttribute("stroke-width", "1.5");
    svg.appendChild(polygon);

    // Colored dots at each vertex
    categoryScores.forEach((val, i) => {
        const r = val * scale;
        const dot = document.createElementNS(ns, "circle");
        dot.setAttribute("cx", px(i, r));
        dot.setAttribute("cy", py(i, r));
        dot.setAttribute("r", size > 100 ? 3.5 : 2);
        dot.setAttribute("fill", RADAR_CATEGORIES[i].color);
        dot.setAttribute("stroke", "#fff");
        dot.setAttribute("stroke-width", "0.5");
        svg.appendChild(dot);
    });

    // Labels (only for larger sizes)
    if (showLabels) {
        const labelR = maxR + 18;
        RADAR_CATEGORIES.forEach((cat, i) => {
            const text = document.createElementNS(ns, "text");
            const x = px(i, labelR);
            const y = py(i, labelR);
            text.setAttribute("x", x);
            text.setAttribute("y", y);
            text.setAttribute("text-anchor", "middle");
            text.setAttribute("dominant-baseline", "central");
            text.setAttribute("fill", cat.color);
            text.setAttribute("font-size", "12");
            text.setAttribute("font-weight", "600");
            text.setAttribute("font-family", "system-ui, sans-serif");
            text.textContent = cat.name;
            svg.appendChild(text);
        });
    }

    container.appendChild(svg);
    return svg;
}

/**
 * Render a radar glyph with a labeled legend below it.
 * Suitable for the feature detail page where space is available.
 *
 * @param {HTMLElement} container - DOM element to render into.
 * @param {number[]} categoryScores - 6 normalized category scores.
 * @param {Object} [opts] - Options (passed to renderRadarGlyph, plus extras).
 * @param {number} [opts.size=180] - SVG size.
 */
function renderRadarWithLegend(container, categoryScores, opts = {}) {
    const size = opts.size || 180;
    const wrapper = document.createElement("div");
    wrapper.style.cssText = "display:flex; flex-direction:column; align-items:center; gap:0.5rem;";

    // Radar glyph
    renderRadarGlyph(wrapper, categoryScores, { ...opts, size, showLabels: true });

    container.appendChild(wrapper);
}
