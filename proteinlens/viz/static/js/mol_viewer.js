/**
 * mol_viewer.js — 3Dmol.js wrapper for per-residue activation coloring.
 *
 * Provides functions to:
 *   1. Create a 3Dmol viewer in a container div
 *   2. Load a PDB from /api/pdb/{accession}
 *   3. Color residues white->red by activation value
 *   4. Lazy-load viewers via IntersectionObserver (only init when scrolled into view)
 *
 * WebGL context management:
 *   Browsers limit active WebGL contexts (~8-16). We track active viewers
 *   and dispose the oldest when the limit is reached.
 *
 * Dependencies: 3Dmol.js (loaded via CDN in feature.html)
 */

// Track active viewers for WebGL context management
const _activeViewers = [];
const MAX_WEBGL_CONTEXTS = 8;

/**
 * Create a 3Dmol.js viewer, load a PDB, and color residues by activation.
 *
 * @param {HTMLElement} container      - DOM element (should have fixed width/height, e.g. 400x300).
 * @param {string} accession           - UniProt accession to fetch PDB for.
 * @param {Array<number>} activations  - Per-residue activation values.
 * @param {number} maxActivation       - Feature-level max activation for normalization.
 * @returns {Promise<Object|null>}     - The 3Dmol viewer object, or null on failure.
 */
async function createMolViewer(container, accession, activations, maxActivation) {
    // Enforce WebGL context limit by disposing oldest viewer if at capacity
    if (_activeViewers.length >= MAX_WEBGL_CONTEXTS) {
        const oldest = _activeViewers.shift();
        try {
            oldest.viewer.clear();
            // Remove the canvas from the DOM to release the WebGL context
            const oldCanvas = oldest.container.querySelector("canvas");
            if (oldCanvas) oldCanvas.remove();
        } catch (e) {
            // Best-effort cleanup
        }
        // Show a placeholder indicating the viewer was recycled
        oldest.container.innerHTML = '<div class="viewer-placeholder">Scrolled out of view</div>';
    }

    try {
        // Fetch PDB data
        const res = await fetch(`/api/pdb/${accession}`);
        if (!res.ok) {
            container.innerHTML = '<div class="viewer-placeholder">No structure available</div>';
            return null;
        }
        const pdbData = await res.text();

        // Clear container and create viewer
        container.innerHTML = "";
        const viewer = $3Dmol.createViewer(container, {
            backgroundColor: "white",
            antialias: true,
        });

        viewer.addModel(pdbData, "pdb");

        // Color residues by activation: white (0) -> red (max)
        // 3Dmol uses 0-indexed residue serial numbers (resi), but PDB files
        // use 1-based. We iterate activations and set color per residue index.
        const model = viewer.getModel();
        const atoms = model.selectedAtoms({});

        // Build a color map: residue number (1-based in PDB) -> hex integer
        // 3Dmol.js colorfunc expects numeric hex colors (e.g. 0xFF2626)
        const colorMap = {};
        for (let i = 0; i < activations.length; i++) {
            const act = activations[i] || 0;
            const norm = maxActivation > 0 ? Math.min(act / maxActivation, 1) : 0;
            // White (0xFFFFFF) -> Red (0xFF2626)
            const r = 255;
            const g = Math.round(255 - norm * (255 - 38));
            const b = Math.round(255 - norm * (255 - 38));
            // PDB residue numbering is 1-based
            colorMap[i + 1] = (r << 16) | (g << 8) | b;
        }

        // Apply cartoon style with per-residue coloring
        viewer.setStyle({}, {
            cartoon: {
                colorfunc: function (atom) {
                    return colorMap[atom.resi] ?? 0xFFFFFF;
                },
            },
        });

        viewer.zoomTo();
        viewer.render();

        // Track for WebGL management
        _activeViewers.push({ viewer, container });

        return viewer;
    } catch (err) {
        console.error(`Failed to create viewer for ${accession}:`, err);
        container.innerHTML = '<div class="viewer-placeholder">Viewer error</div>';
        return null;
    }
}

/**
 * Set up lazy loading for a viewer container using IntersectionObserver.
 *
 * The viewer will only be initialized when the container scrolls into the
 * viewport (or when a parent <details> element is opened).
 *
 * @param {HTMLElement} container      - DOM element for the viewer (400x300).
 * @param {string} accession           - UniProt accession.
 * @param {Array<number>} activations  - Per-residue activation values.
 * @param {number} maxActivation       - Feature-level max activation.
 */
function lazyLoadViewer(container, accession, activations, maxActivation) {
    // Show placeholder initially
    container.innerHTML = '<div class="viewer-placeholder">Scroll to load 3D structure</div>';

    // Mark container with data attributes for re-initialization if needed
    container.dataset.accession = accession;
    container.dataset.loaded = "false";

    const observer = new IntersectionObserver(
        (entries) => {
            for (const entry of entries) {
                if (entry.isIntersecting && container.dataset.loaded === "false") {
                    container.dataset.loaded = "true";
                    container.innerHTML = '<div class="viewer-placeholder"><div class="loading-spinner"></div> Loading structure...</div>';
                    createMolViewer(container, accession, activations, maxActivation);
                    observer.unobserve(container);
                }
            }
        },
        { rootMargin: "200px" } // Start loading slightly before visible
    );

    observer.observe(container);
}

/**
 * Create a 3Dmol viewer for a motif superposition structure.
 *
 * Colors residues by per_position_flexibility: blue (rigid, 0) -> red (flexible, max).
 *
 * @param {HTMLElement} container         - DOM element for the viewer.
 * @param {string} pdbString             - PDB-format string of the mean motif structure.
 * @param {Array<number>} flexibility    - Per-position flexibility values.
 */
function createMotifViewer(container, pdbString, flexibility) {
    if (!pdbString) {
        container.innerHTML = '<div class="viewer-placeholder">No motif structure</div>';
        return;
    }

    container.innerHTML = "";
    const viewer = $3Dmol.createViewer(container, {
        backgroundColor: "white",
        antialias: true,
    });

    viewer.addModel(pdbString, "pdb");

    // Color by flexibility: blue (rigid) -> red (flexible)
    const maxFlex = Math.max(...flexibility.filter((v) => typeof v === "number"), 1);
    const colorMap = {};
    for (let i = 0; i < flexibility.length; i++) {
        const val = typeof flexibility[i] === "number" ? flexibility[i] : 0;
        const norm = maxFlex > 0 ? Math.min(val / maxFlex, 1) : 0;
        // Blue (0x0000FF) -> Red (0xFF0000)
        const r = Math.round(norm * 255);
        const g = 0;
        const b = Math.round((1 - norm) * 255);
        colorMap[i + 1] = (r << 16) | (g << 8) | b;
    }

    viewer.setStyle({}, {
        cartoon: {
            colorfunc: function (atom) {
                return colorMap[atom.resi] ?? 0x808080;
            },
        },
    });

    viewer.zoomTo();
    viewer.render();

    _activeViewers.push({ viewer, container });
}
