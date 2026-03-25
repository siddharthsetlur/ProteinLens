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


// ============================================================
// Text-based sequence display (AA letters with colored backgrounds)
// ============================================================

/**
 * Create a text-based sequence display where each amino acid letter is shown
 * in a monospace font with a background color proportional to its activation.
 *
 * This produces the "alignment view" seen in MSA tools: a row of characters
 * where color encodes a per-residue value (here, SAE activation).
 *
 * @param {HTMLElement} container       - DOM element to append the sequence row to.
 * @param {Object} opts                 - Configuration object.
 * @param {string} opts.sequence        - Amino acid sequence string.
 * @param {Array<number>} opts.activations - Per-residue activation values.
 * @param {number} opts.maxActivation   - Global max activation for normalization.
 * @param {string} [opts.accession]     - UniProt accession (shown in label).
 * @param {number} [opts.maxAct]        - Per-protein max activation (shown in label).
 * @param {boolean} [opts.showLabel=true] - Whether to show the accession label.
 * @param {string} [opts.cssClass]      - Optional extra CSS class for the wrapper.
 */
function createTextSequence(container, opts) {
    const {
        sequence,
        activations,
        maxActivation,
        accession = "",
        maxAct = null,
        showLabel = true,
        cssClass = "",
    } = opts;

    const row = document.createElement("div");
    row.className = "seq-row";

    // Label column (accession + copy button)
    if (showLabel) {
        const label = document.createElement("div");
        label.className = "seq-row-label";

        const copyBtn = document.createElement("button");
        copyBtn.className = "copy-btn";
        copyBtn.textContent = "\u{1F4CB}";  // clipboard emoji
        copyBtn.title = "Copy accession";
        copyBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            navigator.clipboard.writeText(accession);
        });
        label.appendChild(copyBtn);

        const nameSpan = document.createElement("span");
        nameSpan.textContent = accession;
        nameSpan.title = maxAct !== null
            ? `${accession} · max: ${maxAct.toFixed(3)}`
            : accession;
        label.appendChild(nameSpan);

        row.appendChild(label);
    }

    // Sequence letters — each AA is a <span> with a colored background
    const lettersDiv = document.createElement("div");
    lettersDiv.className = "seq-letters";

    for (let i = 0; i < sequence.length; i++) {
        const span = document.createElement("span");
        span.className = "aa";
        span.textContent = sequence[i];

        const act = activations[i] || 0;
        const norm = maxActivation > 0 ? act / maxActivation : 0;
        span.style.backgroundColor = activationToColor(norm);

        // Dark text for light backgrounds, white text for dark (high activation)
        if (norm > 0.6) {
            span.style.color = "#fff";
        }

        // Tooltip on hover: residue letter, 1-based position, activation value
        span.title = `${sequence[i]}${i + 1}: ${act.toFixed(3)}`;

        lettersDiv.appendChild(span);
    }

    row.appendChild(lettersDiv);

    const wrapper = document.createElement("div");
    wrapper.className = cssClass ? `seq-text-inline ${cssClass}` : "seq-text-inline";
    wrapper.appendChild(row);
    container.appendChild(wrapper);
}

/**
 * Find the anchor position for a protein's activation profile.
 *
 * "first" mode: index of the first residue with activation above 5% of the
 *               feature max (i.e. the first meaningfully activated residue).
 * "max" mode:   index of the residue with the highest activation value.
 *
 * @param {Array<number>} activations  - Per-residue activation values.
 * @param {number} maxActivation       - Feature-level max activation.
 * @param {string} mode                - "first" or "max".
 * @returns {number} 0-based index of the anchor residue, or 0 if none found.
 */
function _findAnchor(activations, maxActivation, mode) {
    const threshold = maxActivation * 0.05;

    if (mode === "max") {
        let bestIdx = 0;
        let bestVal = -1;
        for (let i = 0; i < activations.length; i++) {
            if ((activations[i] || 0) > bestVal) {
                bestVal = activations[i] || 0;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    // mode === "first": first residue above threshold
    for (let i = 0; i < activations.length; i++) {
        if ((activations[i] || 0) > threshold) return i;
    }
    return 0;
}

/**
 * Render a single aligned sequence row with left-padding dashes.
 *
 * The sequence is shifted right by `padLeft` dash characters so that the
 * anchor position aligns vertically with other sequences in the block.
 * Dashes are rendered as dim gray characters with no activation color.
 *
 * @param {HTMLElement} container       - Parent element to append the row to.
 * @param {Object} protein              - Protein data with sequence + activations.
 * @param {number} maxActivation        - Feature-level max activation for color normalization.
 * @param {number} padLeft              - Number of dash characters to prepend.
 */
function _renderAlignedRow(container, protein, maxActivation, padLeft) {
    const seq = protein.sequence || "";
    const acts = protein.per_residue_activations || [];

    const row = document.createElement("div");
    row.className = "seq-row";

    // Label with copy button
    const label = document.createElement("div");
    label.className = "seq-row-label";

    const copyBtn = document.createElement("button");
    copyBtn.className = "copy-btn";
    copyBtn.textContent = "\u{1F4CB}";
    copyBtn.title = "Copy accession";
    copyBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        navigator.clipboard.writeText(protein.accession || "");
    });
    label.appendChild(copyBtn);

    const nameSpan = document.createElement("span");
    nameSpan.textContent = protein.accession || "";
    nameSpan.title = `${protein.accession} · max: ${(protein.max_activation || 0).toFixed(3)}`;
    label.appendChild(nameSpan);
    row.appendChild(label);

    // Sequence letters with leading dashes for alignment
    const lettersDiv = document.createElement("div");
    lettersDiv.className = "seq-letters";

    // Leading dashes (gap characters)
    for (let i = 0; i < padLeft; i++) {
        const dash = document.createElement("span");
        dash.className = "aa aa-gap";
        dash.textContent = "-";
        lettersDiv.appendChild(dash);
    }

    // Actual residues with activation coloring
    for (let i = 0; i < seq.length; i++) {
        const span = document.createElement("span");
        span.className = "aa";
        span.textContent = seq[i];

        const act = acts[i] || 0;
        const norm = maxActivation > 0 ? act / maxActivation : 0;
        span.style.backgroundColor = activationToColor(norm);

        if (norm > 0.6) {
            span.style.color = "#fff";
        }

        span.title = `${seq[i]}${i + 1}: ${act.toFixed(3)}`;
        lettersDiv.appendChild(span);
    }

    row.appendChild(lettersDiv);
    container.appendChild(row);
}

/**
 * Create a multi-sequence alignment view with anchor-based alignment.
 *
 * Sequences are horizontally shifted so that a chosen anchor position
 * (first activation or max activation) lines up vertically across all rows.
 * Leading positions before the sequence start are shown as dashes.
 *
 * A toggle control lets the user switch between "first activation" and
 * "max activation" alignment modes. Default is "first activation".
 *
 * @param {HTMLElement} container       - DOM element to append the alignment block to.
 * @param {Array<Object>} proteins      - Array of protein objects from top_sequences.
 * @param {number} maxActivation        - Feature-level max activation for normalization.
 * @param {number} [topN=8]             - Number of proteins to show.
 */
function createAlignmentView(container, proteins, maxActivation, topN = 8) {
    const toShow = proteins.slice(0, topN);
    if (toShow.length === 0) return;

    // --- Mode toggle control ---
    const controls = document.createElement("div");
    controls.style.cssText = "margin-bottom:0.5rem; font-size:0.85rem; display:flex; align-items:center; gap:0.75rem;";
    controls.innerHTML = `
        <span style="font-weight:600; color:var(--pico-muted-color);">Align by:</span>
        <label style="cursor:pointer"><input type="radio" name="align-mode" value="first" checked> First activation</label>
        <label style="cursor:pointer"><input type="radio" name="align-mode" value="max"> Max activation</label>
    `;
    container.appendChild(controls);

    // --- Alignment block (re-rendered on mode change) ---
    const block = document.createElement("div");
    block.className = "seq-alignment";
    container.appendChild(block);

    function render(mode) {
        block.innerHTML = "";

        // Compute anchor position for each protein
        const anchors = toShow.map((p) =>
            _findAnchor(p.per_residue_activations || [], maxActivation, mode)
        );

        // The maximum anchor offset determines how much left-padding each row needs.
        // We shift every sequence right so that the largest anchor lines up.
        const maxAnchor = Math.max(...anchors);

        for (let i = 0; i < toShow.length; i++) {
            const padLeft = maxAnchor - anchors[i];
            _renderAlignedRow(block, toShow[i], maxActivation, padLeft);
        }
    }

    // Initial render with default mode
    render("first");

    // Toggle handler
    controls.addEventListener("change", (e) => {
        if (e.target.name === "align-mode") {
            render(e.target.value);
        }
    });
}
