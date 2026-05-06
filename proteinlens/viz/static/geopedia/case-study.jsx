// Case studies — three layer-scoped pages, all wired to live /api endpoints.
// Case 01: Geometry annotates features missing DB labels (cross_family_geometry.json)
// Case 02: Geometry is more granular than biology (subdomain_case_study.json)
// Case 03: Transfer to metagenomic proteins (nmpfam_case_study.json)

// ────────────────────────────────────────────────────────────────────────
// Case study 01 — DB-silent, geometry-significant features.
// ────────────────────────────────────────────────────────────────────────
function CaseStudyGeometry({ layer, onPickFeature }) {
  const cs = useFetch(() => API.crossFamilyGeometry(layer), [layer]);

  if (cs.loading) return <div className="container"><Loading what="case study 01" /></div>;
  if (cs.error)   return <div className="container"><ErrorBox err={cs.error} title="Failed to load case study 01" /></div>;
  const data = cs.data || {};
  const stats = data.global_stats || {};
  const feats = data.features || [];

  return (
    <div className="container">
      <div style={{ padding: '24px 0 12px' }}>
        <span className="eyebrow">Case study · 01 · Layer {layer}</span>
        <h2 style={{ fontFamily: 'var(--serif)', fontSize: 40, fontWeight: 500, letterSpacing: '-0.02em', margin: '8px 0 12px', lineHeight: 1.05 }}>
          Geometry annotates features missing DB labels
        </h2>
        <p style={{ fontFamily: 'var(--serif)', fontSize: 16, color: 'var(--ink-2)', maxWidth: '60ch', lineHeight: 1.5, fontStyle: 'italic' }}>
          Features whose four database / sequence methods (InterPro residue + protein, CATH residue, MEME motif,
          sequence position) are all <em>not</em> significant, but where the geometric Cα classifier reaches
          BH q&nbsp;&lt;&nbsp;0.05. This is the population the paper calls "geometry-primary".
        </p>
      </div>

      <div className="cs-stat-tiles" style={{ margin: '24px 0' }}>
        <div className="cs-stat-tile">
          <div className="lbl">Geometry-primary</div>
          <div className="v geom">{(stats.total_geometry_primary || 0).toLocaleString()}</div>
          <div className="sub">features at q &lt; 0.05 with all four DB / sequence methods n.s.</div>
        </div>
        <div className="cs-stat-tile">
          <div className="lbl">All features with geometric eval</div>
          <div className="v">{(stats.n_features_with_geometry || 0).toLocaleString()}</div>
          <div className="sub">denominator (features that received the geometric test)</div>
        </div>
        <div className="cs-stat-tile">
          <div className="lbl">Cross-family</div>
          <div className="v geom">{stats.n_cross_family != null ? stats.n_cross_family : '—'}</div>
          <div className="sub">{stats.pct_cross_family != null ? `${stats.pct_cross_family.toFixed(1)}% of geometry-primary span ≥ 2 InterPro families` : ''}</div>
        </div>
      </div>

      <div className="section-head" style={{ marginTop: 36 }}>
        <div>
          <span className="eyebrow">Features</span>
          <h3>Geometry-primary feature list</h3>
        </div>
        <p className="desc">
          Sorted by composite score. Click a row to open the full feature page.
        </p>
      </div>

      <div className="cs-rows">
        {[...feats].sort((a, b) => (b.composite_score || 0) - (a.composite_score || 0)).slice(0, 200).map((f) => (
          <button
            key={f.feature_id}
            onClick={() => onPickFeature(layer, f.feature_id)}
            className="cs-row-btn">
            <span className="code">f/{f.feature_id}</span>
            <div style={{ minWidth: 0 }}>
              <div className="name">{f.structural_category || f.top_geometric_feature || "Geometry-primary feature"}</div>
              <div className="sub">
                GBM CV AUC {fmt(f.gbm_auc_cv, 3)}
                {f.concordance_prauc != null ? ` · concordance PR-AUC ${fmt(f.concordance_prauc, 3)}` : ''}
                {f.is_cross_family ? ' · cross-family' : ''}
                {f.n_families_above_05 ? ` · ${f.n_families_above_05} families F1>0.5` : ''}
              </div>
            </div>
            <span className="pill-yes">geom</span>
            <span className="pill-no">db n.s.</span>
            <span style={{ color: 'var(--ink-3)', fontFamily: 'var(--mono)', fontSize: 12 }}>→</span>
          </button>
        ))}
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────
// Case study 02 — same residue-level annotation, split by geometry.
// ────────────────────────────────────────────────────────────────────────
function CaseStudyGranularity({ layer, onPickFeature, onPickFamily }) {
  const cs = useFetch(() => API.subdomain(layer), [layer]);
  const [tab, setTab] = React.useState("interpro");

  if (cs.loading) return <div className="container"><Loading what="case study 02" /></div>;
  if (cs.error)   return <div className="container"><ErrorBox err={cs.error} title="Failed to load case study 02" /></div>;

  const data = cs.data || {};
  const stats = data.global_stats || {};
  const groups = (tab === "interpro" ? data.interpro_groups : data.cath_groups) || [];

  return (
    <div className="container">
      <div style={{ padding: '24px 0 12px' }}>
        <span className="eyebrow">Case study · 02 · Layer {layer}</span>
        <h2 style={{ fontFamily: 'var(--serif)', fontSize: 40, fontWeight: 500, letterSpacing: '-0.02em', margin: '8px 0 12px', lineHeight: 1.05 }}>
          Geometry is more granular than biology
        </h2>
        <p style={{ fontFamily: 'var(--serif)', fontSize: 16, color: 'var(--ink-2)', maxWidth: '60ch', lineHeight: 1.5, fontStyle: 'italic' }}>
          Groups of features that share an InterPro-residue or CATH-residue annotation, but split into distinct
          geometric sub-signatures (mean pairwise cosine of the 44-dim importance vector &lt; 0.5). Where the
          database says one thing, geometry can resolve sub-structure.
        </p>
      </div>

      <div className="cs-stat-tiles" style={{ margin: '24px 0' }}>
        <div className="cs-stat-tile">
          <div className="lbl">InterPro-Res shared groups</div>
          <div className="v">{stats.n_interpro_groups != null ? stats.n_interpro_groups : '—'}</div>
          <div className="sub">{stats.n_interpro_features_in_groups != null ? `${stats.n_interpro_features_in_groups.toLocaleString()} features in groups` : ''}</div>
        </div>
        <div className="cs-stat-tile">
          <div className="lbl">CATH-Res shared groups</div>
          <div className="v">{stats.n_cath_groups != null ? stats.n_cath_groups : '—'}</div>
          <div className="sub">{stats.n_cath_features_in_groups != null ? `${stats.n_cath_features_in_groups.toLocaleString()} features in groups` : ''}</div>
        </div>
        <div className="cs-stat-tile">
          <div className="lbl">Sparsity gate</div>
          <div className="v">{stats.max_pct_activated != null ? `${stats.max_pct_activated}%` : '—'}</div>
          <div className="sub">max % proteins per feature · q &lt; {stats.q_gate ?? '0.05'}</div>
        </div>
      </div>

      <div className="layer-controls" style={{ marginTop: 24 }}>
        <div className="filter-group">
          <span className="filter-label">Source</span>
          <div className="seg">
            <button className={tab === "interpro" ? "on" : ""} onClick={() => setTab("interpro")}>InterPro-Res</button>
            <button className={tab === "cath"     ? "on" : ""} onClick={() => setTab("cath")}>CATH-Res</button>
          </div>
        </div>
        <span className="row-count">{groups.length.toLocaleString()} groups</span>
      </div>

      <div className="cs-rows" style={{ marginTop: 0 }}>
        {groups.map((g) => (
          <CS2GroupRow
            key={g.annotation_code || g.code}
            g={g}
            source={tab}
            onPick={() => onPickFamily && onPickFamily(tab, g.annotation_code || g.code)}
          />
        ))}
      </div>
    </div>
  );
}

function CS2GroupRow({ g, source, onPick }) {
  const cosine = g.mean_cosine_similarity ?? g.mean_cos;
  const f1 = source === "interpro" ? (g.mean_residue_f1 ?? g.ipr_f1) : (g.mean_residue_f1 ?? g.cath_f1);
  return (
    <button className="cs-row-btn" onClick={onPick}>
      <span className="code">{g.annotation_code || g.code}</span>
      <div style={{ minWidth: 0 }}>
        <div className="name">{g.annotation_name || g.name}</div>
        <div className="sub">
          {g.n_features != null ? `${g.n_features} features` : ''}
          {f1 != null ? ` · ${source === "interpro" ? "ipr" : "cath"}-res F1 ${fmt(f1, 2)}` : ''}
          {cosine != null ? ` · mean cos ${fmt(cosine, 2)}` : ''}
        </div>
      </div>
      <span className={cosine != null && cosine < 0.5 ? "pill-yes" : "pill-no"}>geom split</span>
      <span className="pill-no">{g.n_distinct_categories || g.n_distinct_top_features || '—'} distinct</span>
      <span style={{ color: 'var(--ink-3)', fontFamily: 'var(--mono)', fontSize: 12 }}>→</span>
    </button>
  );
}

// ────────────────────────────────────────────────────────────────────────
// Case study 03 — NMPFam metagenomic transfer (paper §4.3 + Table 4 + Fig. 4).
//
// Data flow:
//   /api/layers/{L}/nmpfam-transfer-summary returns
//     { table4: {...}, features: [{feature_id, max_prauc, median_prauc,
//         n_hits, n_strong, sequences_annotated, top_hits[]} ...] }
//   The features list is *already gated* server-side to Table 4 column 3
//   (geometry q-significant AND median per-family PR-AUC > 0.5) and sorted
//   by max_prauc descending. We just render it.
// ────────────────────────────────────────────────────────────────────────
function CaseStudyMetagenomic({ layer, onPickFeature }) {
  const cs = useFetch(
    () => API.nmpfamTransferSummary(layer).catch(e => ({ __error: e })),
    [layer]
  );
  const [sortKey, setSortKey] = React.useState("max_prauc");

  if (cs.loading) return <div className="container"><Loading what="case study 03 transfer summary" /></div>;
  const data = cs.data || {};
  if (data.__error) {
    return (
      <div className="container">
        <div style={{ padding: '24px 0 12px' }}>
          <span className="eyebrow">Case study · 03 · Layer {layer}</span>
          <h2 style={{ fontFamily: 'var(--serif)', fontSize: 40, fontWeight: 500, letterSpacing: '-0.02em', margin: '8px 0 12px', lineHeight: 1.05 }}>
            Transfer to metagenomic proteins
          </h2>
          <div className="error-box">
            <strong>Not built for layer {layer}.</strong>
            <div style={{ marginTop: 4 }}>
              Run <code>python scripts/build_nmpfam_transfer_summary.py --analysis-dir analysis/l{layer}</code>
              {' '}to generate the per-feature transfer aggregates.
            </div>
          </div>
        </div>
      </div>
    );
  }

  const t = data.table4 || {};
  const features = data.features || [];
  const sorted = React.useMemo(
    () => [...features].sort((a, b) => (b[sortKey] || 0) - (a[sortKey] || 0)),
    [features, sortKey]
  );

  return (
    <div className="container">
      <div style={{ padding: '24px 0 12px' }}>
        <span className="eyebrow">Case study · 03 · Layer {layer} · §4.3</span>
        <h2 style={{ fontFamily: 'var(--serif)', fontSize: 40, fontWeight: 500, letterSpacing: '-0.02em', margin: '8px 0 12px', lineHeight: 1.05 }}>
          Geometric annotation transfers to metagenomic proteins
        </h2>
        <p style={{ fontFamily: 'var(--serif)', fontSize: 16, color: 'var(--ink-2)', maxWidth: '60ch', lineHeight: 1.5, fontStyle: 'italic' }}>
          For every SAE feature that fires on NMPFams metagenomic clusters, we run the pre-trained
          Swiss-Prot geometric classifier on each metagenomic protein and report PR-AUC. A feature
          whose <strong>median per-family PR-AUC &gt; 0.5</strong> indicates the geometric annotation
          generalises beyond the training distribution.
        </p>
      </div>

      {/* Table 4 stats — exactly the columns from the paper */}
      <CS3Table4Panel t={t} />

      <div className="section-head" style={{ marginTop: 32 }}>
        <div>
          <span className="eyebrow">Transferring features</span>
          <h3>{features.length.toLocaleString()} features pass the Table 4 column 3 gate</h3>
        </div>
        <p className="desc">
          Geometry q&nbsp;&lt;&nbsp;0.05 AND median per-family PR-AUC &gt; 0.5.
          Default sort by max PR-AUC. Click a row to open the per-feature transfer view.
        </p>
      </div>

      <div className="layer-controls" style={{ marginTop: 4 }}>
        <div className="filter-group">
          <span className="filter-label">Sort</span>
          <div className="seg">
            {[
              ["max_prauc",          "max PR-AUC"],
              ["median_prauc",       "median PR-AUC"],
              ["n_strong",           "# strong hits"],
              ["sequences_annotated","sequences annotated"],
              ["n_hits",             "total hits"],
            ].map(([v, l]) => (
              <button key={v} className={sortKey === v ? "on" : ""} onClick={() => setSortKey(v)}>{l}</button>
            ))}
          </div>
        </div>
        <span className="row-count">
          showing top {Math.min(200, sorted.length).toLocaleString()} of {sorted.length.toLocaleString()}
        </span>
      </div>

      <CS3FeatureTable
        rows={sorted.slice(0, 200)}
        onPickFeature={(fid) => onPickFeature(layer, fid)}
      />
    </div>
  );
}

// Per-layer Table 4 reproduction.
function CS3Table4Panel({ t }) {
  const cells = [
    { lbl: "% NMPFam DB activation",
      v: t.pct_with_nmpfam_hits != null ? `${fmt(t.pct_with_nmpfam_hits, 2)}%` : '—',
      sub: t.n_with_nmpfam_hits != null ? `${t.n_with_nmpfam_hits.toLocaleString()} of ${t.n_features_total?.toLocaleString() || '—'} features fire` : '' },
    { lbl: "% q-significant",
      v: t.pct_qsig_of_with_hits != null ? `${fmt(t.pct_qsig_of_with_hits, 2)}%` : '—',
      sub: t.n_qsig_with_hits != null ? `${t.n_qsig_with_hits.toLocaleString()} also geom q < ${t.q_gate ?? '0.05'}` : '' },
    { lbl: "% feat. median PR-AUC > 0.5",
      v: t.pct_features_median_above_gate != null ? `${fmt(t.pct_features_median_above_gate, 2)}%` : '—',
      sub: t.n_features_median_prauc_above_gate != null ? `${t.n_features_median_prauc_above_gate.toLocaleString()} features (gate: ${t.prauc_gate ?? 0.5})` : '',
      kind: "geom" },
    { lbl: "NMPFams matched",
      v: t.n_families_matched != null ? t.n_families_matched.toLocaleString() : '—',
      sub: 'distinct families with PR-AUC > 0.5 from any feature' },
    { lbl: "Sequences annotated",
      v: t.n_sequences_annotated != null ? t.n_sequences_annotated.toLocaleString() : '—',
      sub: 'union sequence_count across matched families' },
  ];
  return (
    <div className="cs-table4">
      {cells.map((c, i) => (
        <div key={i} className={"cs-table4-cell" + (c.kind === "geom" ? " geom" : "")}>
          <div className="lbl">{c.lbl}</div>
          <div className="v">{c.v}</div>
          {c.sub && <div className="sub">{c.sub}</div>}
        </div>
      ))}
    </div>
  );
}

// Compact table of gated features. Each row is one feature; columns are
// the per-feature transfer aggregates from the precompute step.
function CS3FeatureTable({ rows, onPickFeature }) {
  if (!rows.length) {
    return (
      <div className="panel" style={{ padding: 14, fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--ink-3)' }}>
        No features pass the Table 4 column 3 gate for this layer.
      </div>
    );
  }
  return (
    <div className="ftable-wrap" style={{ borderTop: '1px solid var(--rule)' }}>
      <table className="ftable">
        <thead>
          <tr>
            <th>ID</th>
            <th title="Total NMPFam family hits with definable PR-AUC">Hits</th>
            <th title="# hits with PR-AUC > 0.5">Strong</th>
            <th>Max PR-AUC</th>
            <th>Median PR-AUC</th>
            <th>Sequences annotated</th>
            <th>Top family · max PR-AUC</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const top = r.top_hits?.[0];
            return (
              <tr key={r.feature_id} onClick={() => onPickFeature(r.feature_id)}>
                <td className="id-cell">f/{r.feature_id}</td>
                <td>{r.n_hits.toLocaleString()}</td>
                <td className="sig geom">{r.n_strong.toLocaleString()}</td>
                <td className="sig geom">{fmt(r.max_prauc, 3)}</td>
                <td>{fmt(r.median_prauc, 3)}</td>
                <td>{(r.sequences_annotated || 0).toLocaleString()}</td>
                <td className="label-cell">
                  <span className="lbl-text" title={top?.family_id || ''}>
                    {top ? `${top.family_id} · ${fmt(top.prauc, 3)} · ${(top.sequence_count || 0).toLocaleString()} seq` : '—'}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────
// Case study 02 — family detail page
// Drills into a single shared-annotation group: lists the member features,
// shows the per-member 44-dim importance heatmap, the pairwise cosine matrix
// from the case study payload, and a shared-protein activation overlay
// where each covering feature's SAE activation (solid) and geom probability
// (dashed) are plotted together.
// ────────────────────────────────────────────────────────────────────────

// Stable column order for the 44-dim Cα descriptor importance heatmap.
const CS2_DESCRIPTORS = [
  "curvature_mean", "curvature_max", "curvature_std",
  "curv_N_third", "curv_centre_third", "curv_C_third",
  "narrow_curvature_mean", "narrow_curvature_max",
  "wide_curvature_mean", "wide_curvature_max",
  "torsion_mean", "torsion_std", "torsion_frac_pos",
  "tors_N_third", "tors_centre_third", "tors_C_third",
  "narrow_torsion_mean", "narrow_torsion_std",
  "wide_torsion_mean", "wide_torsion_std",
  "planarity_mean", "planarity_std",
  "plan_N_third", "plan_centre_third", "plan_C_third",
  "tangent_alignment", "end_to_end_ratio",
  "narrow_tangent_alignment", "narrow_end_to_end_ratio",
  "wide_tangent_alignment", "wide_end_to_end_ratio",
  "contact_density_8A", "contact_density_12A",
  "long_range_contacts_8A", "long_range_contacts_12A",
  "max_seq_sep_contact_8A", "mean_seq_sep_contact_8A",
  "contact_order_local", "min_spatial_dist_long",
  "frac_hydrophobic", "frac_charged", "frac_polar",
  "frac_gly_pro", "frac_aromatic",
];

function CaseStudyFamilyDetail({ layer, family, onBack, onPickFeature }) {
  // Pull the full subdomain payload to find the matching group.
  const cs = useFetch(() => API.subdomain(layer), [layer]);

  const group = React.useMemo(() => {
    if (!cs.data) return null;
    const list = family.source === "interpro"
      ? (cs.data.interpro_groups || [])
      : (cs.data.cath_groups || []);
    return list.find(g => (g.annotation_code || g.code) === family.code) || null;
  }, [cs.data, family]);

  // Fetch geometry_enrichment for every member feature in parallel — this gives
  // us the full 44-dim importance vector and the per-protein activation +
  // geom_prob_profile traces needed for the shared-protein overlay.
  const memberIds = group ? (group.features || []).map(m => m.feature_id) : [];
  const geomDetails = useFetch(
    () => Promise.all(memberIds.map(fid =>
      API.geometry(layer, fid).catch(() => null)
    )),
    [layer, JSON.stringify(memberIds)]
  );

  if (cs.loading) return <div className="container"><Loading what="case study 02" /></div>;
  if (cs.error)   return <div className="container"><ErrorBox err={cs.error} title="Failed to load subdomain case study" /></div>;
  if (!group) {
    return (
      <div className="container">
        <div className="error-box">
          <strong>Family {family.code} not found in layer {layer}.</strong>
        </div>
      </div>
    );
  }

  const cosine = group.mean_cosine_similarity ?? null;
  const features = group.features || [];

  return (
    <div className="container">
      <div style={{ padding: '24px 0 12px' }}>
        <a onClick={onBack} style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--ink-2)', cursor: 'pointer', textDecoration: 'none' }}>
          ← back to family list
        </a>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 14, marginTop: 12 }}>
          <span className="eyebrow">{family.source === "interpro" ? "InterPro residue group" : "CATH residue group"} · Layer {layer}</span>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--ink-3)' }}>{family.code}</span>
        </div>
        <h2 style={{ fontFamily: 'var(--serif)', fontSize: 36, fontWeight: 500, letterSpacing: '-0.02em', margin: '8px 0 12px', lineHeight: 1.05 }}>
          {group.annotation_name || group.name}
        </h2>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--ink-2)' }}>
          {features.length} member features
          {cosine != null ? ` · mean pairwise cosine ${fmt(cosine, 3)}` : ''}
          {cosine != null ? (cosine < 0.5
            ? <span className="pill-yes" style={{ marginLeft: 10 }}>GEOM split (cos &lt; 0.5)</span>
            : <span className="pill-no" style={{ marginLeft: 10 }}>not split by geom</span>) : null}
          {group.mean_geom_pr_auc != null ? ` · mean geom PR-AUC ${fmt(group.mean_geom_pr_auc, 3)}` : ''}
          {group.mean_residue_f1 != null ? ` · mean residue F1 ${fmt(group.mean_residue_f1, 3)}` : ''}
        </div>
      </div>

      <div className="section-head" style={{ marginTop: 32 }}>
        <div>
          <span className="eyebrow">Members</span>
          <h3>Feature list</h3>
        </div>
        <p className="desc">
          Click a feature to open its full page. Top descriptor names and category come from the per-feature geometric classifier.
        </p>
      </div>
      <div className="cs-rows">
        {features.map((m) => (
          <CS2MemberRow
            key={m.feature_id}
            m={m}
            layer={layer}
            onPickFeature={onPickFeature}
          />
        ))}
      </div>

      <div className="section-head" style={{ marginTop: 32 }}>
        <div>
          <span className="eyebrow">Pairwise cosine</span>
          <h3>How similar are these features in the 44-dim importance space?</h3>
        </div>
        <p className="desc">
          Cosine similarity between every pair of member features' GBM importance vectors.
          Lighter cells = members lean on different geometric descriptors.
        </p>
      </div>
      <CS2CosineHeatmap matrix={group.cosine_matrix} ids={features.map(m => m.feature_id)} />

      <div className="section-head" style={{ marginTop: 32 }}>
        <div>
          <span className="eyebrow">44-dim importance</span>
          <h3>Where each member sits in the geometric descriptor space</h3>
        </div>
        <p className="desc">
          Rows are member features, columns are the 44 Cα descriptors used by the geometric GBM.
          A bright row signals a few dominant descriptors; a flat row means importance is spread out.
        </p>
      </div>
      <CS2ImportanceHeatmap features={features} geomDetails={geomDetails} />

      <div className="section-head" style={{ marginTop: 32 }}>
        <div>
          <span className="eyebrow">Shared protein</span>
          <h3>Different members hit different residues on the same protein</h3>
        </div>
        <p className="desc">
          Pick a UniProt accession that ≥ 2 members evaluate on. We overlay each covering feature's
          SAE activation (solid) and geometric probability (dashed) along that protein's residues.
        </p>
      </div>
      <CS2SharedProteinOverlay features={features} geomDetails={geomDetails} />
    </div>
  );
}

function CS2MemberRow({ m, layer, onPickFeature }) {
  return (
    <button className="cs-row-btn" onClick={() => onPickFeature(layer, m.feature_id)}>
      <span className="code">f/{m.feature_id}</span>
      <div style={{ minWidth: 0 }}>
        <div className="name">{m.structural_category || m.top_geometric_feature || "—"}</div>
        <div className="sub">
          {m.top_geometric_feature ? `top descriptor: ${m.top_geometric_feature}` : ''}
          {m.geom_pr_auc != null ? ` · geom PR-AUC ${fmt(m.geom_pr_auc, 3)}` : ''}
          {m.pct_proteins_activated != null ? ` · ${fmt(m.pct_proteins_activated, 2)}% of proteins` : ''}
        </div>
      </div>
      <span className={isSig(m.q_geometry_prauc) ? "pill-yes" : "pill-no"}>geom</span>
      <span className={isSig(m.q_interpro_res_f1) || isSig(m.q_cath_res_f1) ? "pill-yes" : "pill-no"}>res</span>
      <span style={{ color: 'var(--ink-3)', fontFamily: 'var(--mono)', fontSize: 12 }}>→</span>
    </button>
  );
}

// Pairwise cosine matrix as a Plotly heatmap. Falls back to a placeholder
// if the case study payload didn't include cosine_matrix for this group.
function CS2CosineHeatmap({ matrix, ids }) {
  const ref = React.useRef(null);
  React.useEffect(() => {
    if (!ref.current || !window.Plotly) return;
    if (!matrix || !matrix.length) {
      ref.current.innerHTML = '<div class="spinner">No cosine matrix available for this group.</div>';
      return;
    }
    const labels = ids.map(id => `f/${id}`);
    const data = [{
      z: matrix,
      x: labels,
      y: labels,
      type: "heatmap",
      colorscale: "YlOrBr",
      reversescale: true,
      zmin: 0,
      zmax: 1,
      colorbar: { title: "cos", thickness: 12 },
      hovertemplate: "%{y} ↔ %{x}<br>cos %{z:.3f}<extra></extra>",
    }];
    const layout = {
      height: Math.max(360, ids.length * 16 + 120),
      margin: { l: 70, r: 30, t: 10, b: 70 },
      xaxis: { side: "bottom", tickfont: { size: 9, family: "IBM Plex Mono" } },
      yaxis: { autorange: "reversed", tickfont: { size: 9, family: "IBM Plex Mono" } },
      font: { family: "IBM Plex Sans" },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
    };
    Plotly.react(ref.current, data, layout, { displayModeBar: false, responsive: true });
  }, [matrix, JSON.stringify(ids)]);

  return <div ref={ref} className="panel" style={{ padding: 12 }} />;
}

// 44-dim importance heatmap. Pulls feature_importances per member from the
// loaded geometry_enrichment payloads.
function CS2ImportanceHeatmap({ features, geomDetails }) {
  const ref = React.useRef(null);
  React.useEffect(() => {
    if (!ref.current || !window.Plotly) return;
    if (!geomDetails.data) {
      ref.current.innerHTML = '<div class="spinner">Loading per-feature importance vectors…</div>';
      return;
    }
    const z = features.map((m, i) => {
      const det = geomDetails.data[i];
      const importances = det?.geometric_residue_level?.feature_importances || {};
      return CS2_DESCRIPTORS.map(d => Number(importances[d] || 0));
    });
    const data = [{
      z,
      x: CS2_DESCRIPTORS,
      y: features.map(m => `f/${m.feature_id}`),
      type: "heatmap",
      colorscale: "YlOrBr",
      reversescale: true,
      zmin: 0,
      colorbar: { title: "importance", thickness: 12 },
      hovertemplate: "%{y}<br>%{x}<br>%{z:.3f}<extra></extra>",
    }];
    const layout = {
      height: Math.max(360, features.length * 16 + 220),
      margin: { l: 80, r: 30, t: 10, b: 180 },
      xaxis: {
        side: "bottom",
        tickangle: -55,
        tickfont: { size: 8, family: "IBM Plex Mono" },
      },
      yaxis: { autorange: "reversed", tickfont: { size: 9, family: "IBM Plex Mono" } },
      font: { family: "IBM Plex Sans" },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
    };
    Plotly.react(ref.current, data, layout, { displayModeBar: false, responsive: true });
  }, [features, geomDetails.data]);

  return <div ref={ref} className="panel" style={{ padding: 12 }} />;
}

// Shared-protein overlay: pick a UniProt accession that ≥2 members hit
// in their top_proteins; for each covering feature plot its SAE activation
// (solid) and geometric probability (dashed) along that protein's residues.
function CS2SharedProteinOverlay({ features, geomDetails }) {
  const sharedByAccession = React.useMemo(() => {
    if (!geomDetails.data) return [];
    const map = {};   // accession -> [{feature_id, sequence, sae_act, geom_prob}]
    geomDetails.data.forEach((det, i) => {
      const fid = features[i].feature_id;
      const tps = det?.plot_data?.top_proteins || [];
      for (const p of tps) {
        if (!p.accession) continue;
        if (!map[p.accession]) map[p.accession] = [];
        map[p.accession].push({
          feature_id: fid,
          sequence: p.sequence,
          sae_activation_profile: p.sae_activation_profile,
          geom_prob_profile: p.geom_prob_profile,
        });
      }
    });
    return Object.entries(map)
      .filter(([, entries]) => entries.length >= 2)
      .sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]));
  }, [features, geomDetails.data]);

  const [picked, setPicked] = React.useState(null);
  React.useEffect(() => {
    if (sharedByAccession.length && !picked) setPicked(sharedByAccession[0][0]);
  }, [sharedByAccession, picked]);

  const ref = React.useRef(null);
  React.useEffect(() => {
    if (!ref.current || !window.Plotly || !picked) return;
    const entries = (sharedByAccession.find(([a]) => a === picked) || [, []])[1];
    if (!entries.length) return;
    const length = Math.max(...entries.map(e => (e.sae_activation_profile || []).length));
    const x = Array.from({ length }, (_, i) => i + 1);
    const palette = ["#C97A00", "#5C82BD", "#7C7CC4", "#E07333", "#3B6E9B", "#9B6FB7", "#D04848"];
    const traces = [];
    entries.forEach((e, i) => {
      const c = palette[i % palette.length];
      traces.push({
        x, y: e.sae_activation_profile, name: `f/${e.feature_id} · SAE`,
        type: "scatter", mode: "lines",
        line: { color: c, width: 1.5 },
        yaxis: "y",
      });
      traces.push({
        x, y: e.geom_prob_profile, name: `f/${e.feature_id} · Geom`,
        type: "scatter", mode: "lines",
        line: { color: c, width: 1.5, dash: "dash" },
        yaxis: "y2",
      });
    });
    const layout = {
      height: 420,
      margin: { l: 60, r: 60, t: 20, b: 40 },
      xaxis: { title: "Residue", tickfont: { family: "IBM Plex Mono", size: 10 } },
      yaxis: { title: "SAE activation", side: "left", tickfont: { family: "IBM Plex Mono", size: 10 } },
      yaxis2: {
        title: "Geom probability", side: "right", overlaying: "y",
        range: [0, 1], tickfont: { family: "IBM Plex Mono", size: 10 },
      },
      legend: { orientation: "h", x: 0, y: -0.18, font: { size: 10, family: "IBM Plex Mono" } },
      font: { family: "IBM Plex Sans" },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
    };
    Plotly.react(ref.current, traces, layout, { displayModeBar: false, responsive: true });
  }, [picked, sharedByAccession]);

  if (geomDetails.loading) return <div className="panel"><Loading what="member geometry data" /></div>;
  if (!sharedByAccession.length) {
    return (
      <div className="panel" style={{ padding: 14, fontFamily: "var(--mono)", fontSize: 11, color: "var(--ink-3)" }}>
        No protein appears in the top-activating sample of ≥ 2 features in this family.
      </div>
    );
  }

  return (
    <div className="panel" style={{ padding: 14 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginBottom: 8 }}>
        <span className="eyebrow">Shared protein</span>
        <select
          value={picked || ""}
          onChange={(e) => setPicked(e.target.value)}
          style={{ fontFamily: 'var(--mono)', fontSize: 12, padding: '4px 8px', border: '1px solid var(--rule)', background: 'var(--paper)' }}
        >
          {sharedByAccession.map(([acc, entries]) => (
            <option key={acc} value={acc}>
              {acc} — {entries.length} features ({entries.map(e => `f/${e.feature_id}`).join(", ")})
            </option>
          ))}
        </select>
      </div>
      <div ref={ref} />
    </div>
  );
}

window.CaseStudyGeometry      = CaseStudyGeometry;
window.CaseStudyGranularity   = CaseStudyGranularity;
window.CaseStudyMetagenomic   = CaseStudyMetagenomic;
window.CaseStudyFamilyDetail  = CaseStudyFamilyDetail;
