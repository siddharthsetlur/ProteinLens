/**
 * sequence_strip.js — Canvas-based residue activation strip with InterPro domain overlay.
 *
 * Renders a horizontal strip where each residue is a colored rectangle:
 *   - Color: white (0 activation) -> red (max activation)
 *   - Hover tooltip: residue letter, 1-based position, activation value
 *   - Optional: InterPro domain bar below the strip showing domain boundaries
 *
 * Usage:
 *   const container = document.getElementById("my-container");
 *   createSequenceStrip(container, {
 *     sequence: "MKTL...",
 *     activations: [0.0, 0.1, 3.2, ...],
 *     maxActivation: 8.5,            // global max for this feature (for normalization)
 *     accession: "P12345",           // for fetching InterPro domains
 *     bestAnnotationName: "Kinase",  // highlight matching domains (optional)
 *   });
 *
 * Output: Appends a canvas element + tooltip div to the container.
 */

/**
 * Interpolate between white and red based on normalized activation.
 *
 * @param {number} normalizedValue - Value between 0 (no activation) and 1 (max activation).
 * @returns {string} CSS rgb() color string.
 */
function activationToColor(normalizedValue) {
    // Clamp to [0, 1]
    const t = Math.min(Math.max(normalizedValue, 0), 1);
    // White (255,255,255) -> Red (220, 38, 38)
    const r = 255;
    const g = Math.round(255 - t * (255 - 38));
    const b = Math.round(255 - t * (255 - 38));
    return `rgb(${r},${g},${b})`;
}

/**
 * Create a sequence activation strip with hover tooltip.
 *
 * Draws a horizontal canvas where each residue gets a column colored by its
 * activation value (normalized to maxActivation). Appends the canvas and a
 * floating tooltip to the given container.
 *
 * @param {HTMLElement} container       - DOM element to append the strip to.
 * @param {Object} opts                 - Configuration object.
 * @param {string} opts.sequence        - Amino acid sequence string.
 * @param {Array<number>} opts.activations - Per-residue activation values (same length as sequence).
 * @param {number} opts.maxActivation   - Global max activation for this feature (for normalization).
 * @param {string} [opts.accession]     - UniProt accession (for fetching InterPro domain overlay).
 * @param {string} [opts.bestAnnotationName] - InterPro annotation name to highlight (optional).
 */
function createSequenceStrip(container, opts) {
    const { sequence, activations, maxActivation, accession, bestAnnotationName } = opts;
    const seqLen = sequence.length;

    // Create a wrapper div for positioning the tooltip
    const wrapper = document.createElement("div");
    wrapper.className = "sequence-strip-container";

    // --- Main activation strip canvas ---
    const canvas = document.createElement("canvas");
    const stripHeight = 30;
    const domainBarHeight = 14;
    // We'll set actual pixel dimensions based on container width after mount
    canvas.style.width = "100%";
    canvas.style.height = (stripHeight + domainBarHeight + 2) + "px";
    wrapper.appendChild(canvas);

    // --- Tooltip ---
    const tooltip = document.createElement("div");
    tooltip.className = "strip-tooltip";
    wrapper.appendChild(tooltip);

    container.appendChild(wrapper);

    // --- Draw function (called on mount and resize) ---
    function draw() {
        const displayWidth = wrapper.clientWidth;
        if (displayWidth === 0) return; // not yet in DOM

        // Set canvas pixel dimensions (2x for retina clarity)
        const dpr = window.devicePixelRatio || 1;
        const totalHeight = stripHeight + domainBarHeight + 2;
        canvas.width = displayWidth * dpr;
        canvas.height = totalHeight * dpr;
        canvas.style.height = totalHeight + "px";

        const ctx = canvas.getContext("2d");
        ctx.scale(dpr, dpr);

        // Width of each residue column in CSS pixels
        const colWidth = displayWidth / seqLen;

        // Draw activation strip
        for (let i = 0; i < seqLen; i++) {
            const act = activations[i] || 0;
            const norm = maxActivation > 0 ? act / maxActivation : 0;
            ctx.fillStyle = activationToColor(norm);
            ctx.fillRect(i * colWidth, 0, colWidth + 0.5, stripHeight); // +0.5 to avoid gaps
        }

        // Store colWidth for tooltip calculation
        canvas._colWidth = colWidth;
        canvas._seqLen = seqLen;
    }

    // Draw once mounted
    requestAnimationFrame(draw);

    // --- Hover tooltip ---
    canvas.addEventListener("mousemove", (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const colWidth = canvas._colWidth || 1;
        const idx = Math.floor(x / colWidth);
        if (idx < 0 || idx >= seqLen) {
            tooltip.style.display = "none";
            return;
        }
        const residue = sequence[idx] || "?";
        const act = activations[idx] || 0;
        tooltip.textContent = `${residue}${idx + 1}: ${act.toFixed(3)}`;
        tooltip.style.display = "block";

        // Position tooltip near cursor, offset slightly
        tooltip.style.left = Math.min(x + 10, wrapper.clientWidth - 100) + "px";
        tooltip.style.top = "-24px";
    });

    canvas.addEventListener("mouseleave", () => {
        tooltip.style.display = "none";
    });

    // --- InterPro domain overlay (fetched async) ---
    if (accession) {
        fetchAndDrawDomains(canvas, accession, seqLen, stripHeight, bestAnnotationName);
    }
}

/**
 * Fetch InterPro domain data for an accession and draw colored bars
 * below the activation strip.
 *
 * Each domain is drawn as a colored rectangle. Domains matching the
 * bestAnnotationName are drawn in a distinct highlight color (blue).
 * Other domains use a neutral gray.
 *
 * @param {HTMLCanvasElement} canvas - The sequence strip canvas.
 * @param {string} accession        - UniProt accession to fetch domains for.
 * @param {number} seqLen           - Sequence length (for scaling positions).
 * @param {number} yOffset          - Y position to start drawing the domain bar.
 * @param {string} [bestAnnotationName] - Annotation name to highlight.
 */
async function fetchAndDrawDomains(canvas, accession, seqLen, yOffset, bestAnnotationName) {
    try {
        const res = await fetch(`/api/interpro/${accession}`);
        if (!res.ok) return; // No InterPro data available, silently skip

        const data = await res.json();
        const domains = data.domains || [];
        if (domains.length === 0) return;

        const ctx = canvas.getContext("2d");
        const dpr = window.devicePixelRatio || 1;
        const displayWidth = canvas.width / dpr;
        const colWidth = displayWidth / seqLen;
        const barHeight = 10;
        const barY = yOffset + 2;

        // Assign colors: highlight best-matching annotation, gray for others
        // Use a small palette for distinct domain types
        const palette = [
            "#6c757d", "#17a2b8", "#ffc107", "#28a745", "#fd7e14",
            "#6f42c1", "#e83e8c", "#20c997",
        ];
        let colorIdx = 0;
        const nameToColor = {};

        ctx.save();
        ctx.scale(dpr, dpr);

        for (const domain of domains) {
            const name = domain.interpro_name || domain.member_accession || "Unknown";
            const start = (domain.start - 1); // convert 1-based to 0-based
            const end = domain.end;

            // Assign color per annotation name
            if (!(name in nameToColor)) {
                if (bestAnnotationName && name === bestAnnotationName) {
                    nameToColor[name] = "#0d6efd"; // Highlight color (blue)
                } else {
                    nameToColor[name] = palette[colorIdx % palette.length];
                    colorIdx++;
                }
            }

            ctx.fillStyle = nameToColor[name];
            ctx.globalAlpha = 0.7;
            ctx.fillRect(start * colWidth, barY, (end - start) * colWidth, barHeight);
        }

        ctx.restore();
    } catch (err) {
        // Non-critical: domain overlay is optional
        console.warn(`Failed to fetch InterPro domains for ${accession}:`, err);
    }
}
