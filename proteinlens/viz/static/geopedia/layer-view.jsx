// Layer view — feature browser table for a single layer.

function LayerView({ layer, landingSummary, onPickFeature, onOpenCaseStudy, layersAvailable }) {
  const idx = useFetch(() => API.index(layer), [layer]);

  const [filter, setFilter] = React.useState("all");
  const [search, setSearch] = React.useState("");
  const [sortKey, setSortKey] = React.useState("m7_score");
  const [sortDir, setSortDir] = React.useState("desc");

  const all = idx.data || [];

  const rows = React.useMemo(() => {
    let xs = all.filter((r) => {
      const cat = rowCategory(r);
      if (filter !== "all" && cat !== filter) return false;
      if (search) {
        const s = search.toLowerCase();
        const fid = String(r.feature_id);
        if (!fid.includes(s) &&
            !((r.m7_label || "").toLowerCase().includes(s)) &&
            !((r.m1_label || "").toLowerCase().includes(s)) &&
            !((r.m2_label || "").toLowerCase().includes(s)) &&
            !((r.m6_label || "").toLowerCase().includes(s))) return false;
      }
      return true;
    });
    xs = [...xs].sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey];
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "string") return sortDir === "desc" ? bv.localeCompare(av) : av.localeCompare(bv);
      return sortDir === "desc" ? bv - av : av - bv;
    });
    return xs;
  }, [all, filter, search, sortKey, sortDir]);

  // No cap — render all rows. The wrap is overflow:auto so the user
  // scrolls inside the table; the sticky thead keeps headers in view.
  const visible = rows;

  const layerMeta = (landingSummary?.layers || []).find(L => L.layer === layer) || {};

  function sortBy(k) {
    if (sortKey === k) setSortDir(d => d === "desc" ? "asc" : "desc");
    else { setSortKey(k); setSortDir("desc"); }
  }
  const arrow = (k) => sortKey === k ? (sortDir === "desc" ? " ↓" : " ↑") : "";

  if (idx.loading) return <div className="container"><Loading what={`layer ${layer} features`} /></div>;
  if (idx.error)   return <div className="container"><ErrorBox err={idx.error} title={`Failed to load layer ${layer}`} /></div>;

  return (
    <div className="container">
      <div className="layer-header">
        <div>
          <span className="eyebrow">Layer {layer}</span>
          <h2>{(layerMeta.num_features || all.length).toLocaleString()} features in layer {layer}</h2>
          <p className="desc">
            Each row is one SAE feature. Columns show the score under each annotation method;
            the geometric column is the paper's novel test. Highlighted cells are q&nbsp;&lt;&nbsp;0.05.
            Click a row to open the feature.
          </p>
        </div>
        <div className="layer-summary">
          <div><div className="lbl">Annotated (any)</div><div className="v">{fmt(layerMeta.any_pct, 1)}%</div></div>
          <div><div className="lbl">Database only</div><div className="v bio">{fmt((layerMeta.bio_pct ?? 0) - (layerMeta.both_pct ?? 0), 1)}%</div></div>
          <div><div className="lbl">Geometry only</div><div className="v geom">{fmt(layerMeta.geom_pct ? (layerMeta.geom_pct - (layerMeta.both_pct ?? 0)) : null, 1)}%</div></div>
          <div><div className="lbl">Both</div><div className="v" style={{color:'var(--both)'}}>{fmt(layerMeta.both_pct, 1)}%</div></div>
        </div>
      </div>

      {/* Case-study tiles, scoped to this layer */}
      {onOpenCaseStudy && (
        <div style={{
          display: 'flex', gap: 0, margin: '20px 0 4px',
          border: '1px solid var(--rule)', background: 'var(--paper)',
        }}>
          {[
            { id: 'geom-fills-db', n: '01', t: 'Geometry annotates features missing DB labels' },
            { id: 'granularity',   n: '02', t: 'Geometry is more granular than biology' },
            { id: 'metagenomic',   n: '03', t: 'Transfer to metagenomic proteins' },
          ].map((c, i, arr) => (
            <button key={c.id}
              onClick={() => onOpenCaseStudy(c.id)}
              style={{
                flex: 1, textAlign: 'left', padding: '14px 18px',
                borderRightWidth: i !== arr.length - 1 ? 1 : 0,
                borderRightStyle: 'solid', borderRightColor: 'var(--rule)',
                borderTop: 'none', borderBottom: 'none', borderLeft: 'none',
                background: 'transparent', cursor: 'pointer',
                font: 'inherit', color: 'inherit',
              }}>
              <div className="eyebrow" style={{ marginBottom: 6 }}>Case study · {c.n} →</div>
              <div style={{
                fontFamily: 'var(--serif)', fontSize: 15, lineHeight: 1.3,
                fontWeight: 500, color: 'var(--ink)', textWrap: 'pretty',
              }}>{c.t}</div>
            </button>
          ))}
        </div>
      )}

      {/* Controls */}
      <div className="layer-controls">
        <div className="filter-group">
          <span className="filter-label">Significance</span>
          <div className="seg">
            {[
              ["all", "All"],
              ["geom_only", "Geom only"],
              ["bio_only", "Database only"],
              ["both", "Both"],
              ["none", "None"],
            ].map(([v, l]) => (
              <button key={v} className={filter === v ? "on" : ""} onClick={() => setFilter(v)}>{l}</button>
            ))}
          </div>
        </div>
        <div className="filter-group">
          <span className="filter-label">Search</span>
          <div className="search-box">
            <span style={{color:'var(--ink-3)', fontFamily:'var(--mono)', fontSize:'11px'}}>⌕</span>
            <input placeholder="feature id, label, motif…" value={search}
                   onChange={e => setSearch(e.target.value)} />
          </div>
        </div>
        <span className="row-count">
          {rows.length.toLocaleString()} of {all.length.toLocaleString()} features
        </span>
      </div>

      <div className="ftable-wrap">
        <table className="ftable">
          <colgroup>
            <col style={{width: "62px"}} />
            <col style={{width: "44px"}} />
            <col style={{width: "70px"}} />
            <col style={{width: "70px"}} />
            <col style={{width: "70px"}} />
            <col style={{width: "78px"}} />
            <col />
            <col style={{width: "78px"}} />
            <col />
            <col style={{width: "78px"}} />
            <col />
            <col style={{width: "78px"}} />
            <col />
            <col style={{width: "58px"}} />
            <col style={{width: "58px"}} />
            <col style={{width: "58px"}} />
          </colgroup>
          <thead>
            <tr className="group-row">
              <th colSpan="3"></th>
              <th colSpan="2">Activity</th>
              <th colSpan="2" className="geom">Geometric — Cα backbone</th>
              <th colSpan="2" className="bio">MEME motif</th>
              <th colSpan="2" className="bio">InterPro residue</th>
              <th colSpan="2" className="bio">InterPro protein</th>
              <th colSpan="3" className="bio">Compact</th>
            </tr>
            <tr>
              <th onClick={() => sortBy("feature_id")} style={{cursor:'pointer'}}>ID{arrow("feature_id")}</th>
              <th>Profile</th>
              <th>Sig 1·2·3·4·5·6·7</th>
              <th onClick={() => sortBy("pct_proteins_activated")} style={{cursor:'pointer'}}>% prot{arrow("pct_proteins_activated")}</th>
              <th onClick={() => sortBy("max_activation")} style={{cursor:'pointer'}}>Max act{arrow("max_activation")}</th>
              <th className="geom-col" onClick={() => sortBy("m7_score")} style={{cursor:'pointer'}}>PR-AUC{arrow("m7_score")}</th>
              <th className="geom-col">Top geom label</th>
              <th className="bio-col" onClick={() => sortBy("m6_score")} style={{cursor:'pointer'}}>PR-AUC{arrow("m6_score")}</th>
              <th className="bio-col">Top motif</th>
              <th className="bio-col" onClick={() => sortBy("m2_score")} style={{cursor:'pointer'}}>F1{arrow("m2_score")}</th>
              <th className="bio-col">IPR residue label</th>
              <th className="bio-col" onClick={() => sortBy("m1_score")} style={{cursor:'pointer'}}>F1{arrow("m1_score")}</th>
              <th className="bio-col">IPR protein label</th>
              <th className="bio-col center" title="CATH residue F1">CATH·R</th>
              <th className="bio-col center" title="CATH protein F1">CATH·P</th>
              <th className="bio-col center" title="Sequence position F1">Pos</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((r) => {
              const cat = rowCategory(r);
              const radarArr = radarFromObj(r.geometry_radar);
              return (
                <tr key={r.feature_id} onClick={() => onPickFeature(layer, r.feature_id)}>
                  <td className="id-cell">f/{r.feature_id}</td>
                  <td className="radar-cell">
                    {radarArr ? (
                      <RadarGlyph
                        scores={radarArr}
                        size={28}
                        color={cat === "geom_only" || cat === "both" ? "var(--geom)" : "var(--ink-3)"} />
                    ) : null}
                  </td>
                  <td><SigStrip feat={r} /></td>
                  <td>{fmt(r.pct_proteins_activated, 1)}</td>
                  <td>{fmt(r.max_activation, 2)}</td>
                  <td className={"geom-cell " + (isSig(r.m7_q) ? "is-sig" : "")}>
                    <span className={"sig" + (isSig(r.m7_q) ? " geom" : "")}>{fmt(r.m7_score, 2)}</span>
                  </td>
                  <td className={"geom-cell label-cell " + (isSig(r.m7_q) ? "is-sig" : "")}>
                    <span className="lbl-text" title={r.m7_label || ""}>{r.m7_label || "—"}</span>
                  </td>
                  <td className={"bio-cell " + (isSig(r.m6_q) ? "is-sig" : "")}>
                    <span className={"sig" + (isSig(r.m6_q) ? " bio" : "")}>{fmt(r.m6_score, 2)}</span>
                  </td>
                  <td className={"bio-cell label-cell " + (isSig(r.m6_q) ? "is-sig" : "")}>
                    <span className="lbl-text" title={r.m6_label || ""}>{r.m6_label || "—"}</span>
                  </td>
                  <td className={"bio-cell " + (isSig(r.m2_q) ? "is-sig" : "")}>
                    <span className={"sig" + (isSig(r.m2_q) ? " bio" : "")}>{fmt(r.m2_score, 2)}</span>
                  </td>
                  <td className={"bio-cell label-cell " + (isSig(r.m2_q) ? "is-sig" : "")}>
                    <span className="lbl-text" title={r.m2_label || ""}>{r.m2_label || "—"}</span>
                  </td>
                  <td className={"bio-cell " + (isSig(r.m1_q) ? "is-sig" : "")}>
                    <span className={"sig" + (isSig(r.m1_q) ? " bio" : "")}>{fmt(r.m1_score, 2)}</span>
                  </td>
                  <td className={"bio-cell label-cell " + (isSig(r.m1_q) ? "is-sig" : "")}>
                    <span className="lbl-text" title={r.m1_label || ""}>{r.m1_label || "—"}</span>
                  </td>
                  <td className={"center " + (isSig(r.m4_q) ? "bio-cell is-sig" : "dim")}>{fmt(r.m4_score, 2)}</td>
                  <td className={"center " + (isSig(r.m3_q) ? "bio-cell is-sig" : "dim")}>{fmt(r.m3_score, 2)}</td>
                  <td className={"center " + (isSig(r.m5_q) ? "bio-cell is-sig" : "dim")}>{fmt(r.m5_score, 2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

window.LayerView = LayerView;
