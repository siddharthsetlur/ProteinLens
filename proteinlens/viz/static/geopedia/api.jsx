// Thin client for the GeoPedia /api/* endpoints. All routes are layer-namespaced
// except the cross-layer landing payload and a couple of shared resources.

const API = {
  async get(path) {
    const r = await fetch(path, { credentials: "same-origin" });
    if (!r.ok) {
      const text = await r.text().catch(() => "");
      throw new Error(`${r.status} ${r.statusText} — ${path}\n${text.slice(0, 240)}`);
    }
    return r.json();
  },

  // Cross-layer
  layers:        () => API.get("/api/layers"),
  landing:       () => API.get("/api/landing"),
  featured:      () => API.get("/api/featured"),

  // Per-layer
  stats:         (L) => API.get(`/api/layers/${L}/stats`),
  index:         (L) => API.get(`/api/layers/${L}/index`),
  methodCoverage:(L) => API.get(`/api/layers/${L}/method-coverage`),
  feature:       (L, fid) => API.get(`/api/layers/${L}/feature/${fid}`),
  significance:  (L, fid) => API.get(`/api/layers/${L}/feature/${fid}/significance`),
  interpro:      (L, fid) => API.get(`/api/layers/${L}/feature/${fid}/interpro`),
  motif:         (L, fid) => API.get(`/api/layers/${L}/feature/${fid}/motif`),
  position:      (L, fid) => API.get(`/api/layers/${L}/feature/${fid}/position`),
  geometry:      (L, fid) => API.get(`/api/layers/${L}/feature/${fid}/geometry`),
  cath:          (L, fid) => API.get(`/api/layers/${L}/feature/${fid}/cath`),
  nmpfam:        (L, fid) => API.get(`/api/layers/${L}/feature/${fid}/nmpfam`),

  geometryPrimary:    (L) => API.get(`/api/layers/${L}/geometry-primary`),
  crossFamilyGeometry:(L) => API.get(`/api/layers/${L}/cross-family-geometry`),
  subdomain:          (L) => API.get(`/api/layers/${L}/subdomain-case-study`),
  nmpfamCaseStudy:    (L) => API.get(`/api/layers/${L}/nmpfam-case-study`),
  nmpfamTransferSummary: (L) => API.get(`/api/layers/${L}/nmpfam-transfer-summary`),
};

window.API = API;

// Method definitions match index_builder.METHODS. Order is paper order (1..7).
window.METHOD_DEFS = [
  { id: 1, name: "InterPro Protein", short: "IPR Prot", metric: "F1",     kind: "bio"  },
  { id: 2, name: "InterPro Residue", short: "IPR Res",  metric: "F1",     kind: "bio"  },
  { id: 3, name: "CATH Protein",     short: "CATH Prot", metric: "F1",    kind: "bio"  },
  { id: 4, name: "CATH Residue",     short: "CATH Res", metric: "F1",     kind: "bio"  },
  { id: 5, name: "Sequence Position", short: "Position", metric: "F1",    kind: "bio"  },
  { id: 6, name: "Sequence MEME Motif", short: "MEME",  metric: "PR-AUC", kind: "bio"  },
  { id: 7, name: "Geometric",        short: "Geom",     metric: "PR-AUC", kind: "geom" },
];

// Stable category labels for the landing dataset config.
window.SAE_DEFAULTS = {
  model: "ESM-2 8M",
  architecture: "Simple ReLU SAE · 320-d residual stream",
};
