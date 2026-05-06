// Landing page — hero + paper config grid + multi-layer picker + featured features.
// All numbers come from /api/landing and /api/featured (real values from the
// per-layer feature index, no hard-coded mocks).

function Landing({ onPickLayer, onPickFeature }) {
  const landing  = useFetch(API.landing,  []);
  const featured = useFetch(API.featured, []);

  if (landing.loading) return <Loading what="landing data" />;
  if (landing.error)   return <div className="container"><ErrorBox err={landing.error} /></div>;

  const data = landing.data || {};
  const layers = data.layers || [];
  const sae    = data.sae || {};
  const dataset = data.dataset || {};

  // Mean across loaded layers — used as headline numbers under the hero.
  const mean = (key) => {
    const xs = layers.map(L => L[key]).filter(v => v != null);
    return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
  };

  return (
    <React.Fragment>
      {/* HERO */}
      <section className="container hero hero-single">
        <div>
          <span className="eyebrow">GeoPedia · ESM-2 SAE feature atlas</span>
          <h2 className="hero-title">
            Reading the <em>geometry</em> learned by protein language models, one feature at a time.
          </h2>
          <p className="lede">
            For each feature we test seven annotation methods (InterPro · CATH · MEME · sequence position · geometric Cα backbone)
            and report which ones reach BH q&nbsp;&lt;&nbsp;0.05. Pick a layer below to browse, or jump straight to one of the
            features called out in the paper.
          </p>
          <dl className="meta-grid">
            <div><dt>Model</dt><dd>{dataset.esm_model || SAE_DEFAULTS.model}</dd></div>
            <div><dt>Architecture</dt><dd>Simple ReLU SAE · {sae.activation_dim ? `${sae.activation_dim}-d` : ""} residual stream</dd></div>
            <div><dt>Layers in atlas</dt><dd>{layers.map(L => L.layer).join(", ") || "—"}</dd></div>
            <div><dt>Training proteins</dt><dd>{dataset.total_proteins ? dataset.total_proteins.toLocaleString() : "—"}</dd></div>
            <div><dt>Sequence clusters</dt><dd>{dataset.total_clusters ? dataset.total_clusters.toLocaleString() : "—"}</dd></div>
            <div><dt>Features / layer</dt><dd>{layers[0]?.num_features?.toLocaleString() || "—"}</dd></div>
            <div><dt>Mean annotated (any method)</dt><dd>{mean("any_pct") != null ? `${mean("any_pct").toFixed(1)}%` : "—"} · BH q &lt; 0.05</dd></div>
            <div><dt>Mean geometric significant</dt><dd>{mean("geom_pct") != null ? `${mean("geom_pct").toFixed(1)}%` : "—"}</dd></div>
          </dl>
        </div>
      </section>

      {/* LAYER PICKER */}
      <section className="container section">
        <div className="section-head">
          <div>
            <span className="eyebrow">Step 1</span>
            <h3>Pick a layer to explore</h3>
          </div>
          <p className="desc">
            Each ESM-2 transformer layer has its own SAE. Annotation methods are run independently per layer.
            Bars show the % of features each method significantly annotates (BH q&nbsp;&lt;&nbsp;0.05).
          </p>
        </div>
        <div className="layer-grid">
          {layers.map((L) => {
            const rows = [
              { lbl: "% Total annotated",  v: L.total_annotated_pct, cls: "" },
              { lbl: "% Geometric",        v: L.geometric_pct,        cls: "geom" },
              { lbl: "% Seq Motif",        v: L.seq_motif_pct,        cls: "bio" },
              { lbl: "% InterPro Res.",    v: L.interpro_res_pct,     cls: "bio" },
              { lbl: "% InterPro Prot.",   v: L.interpro_prot_pct,    cls: "bio" },
              { lbl: "% CATH Res.",        v: L.cath_res_pct,         cls: "bio" },
              { lbl: "% CATH Prot.",       v: L.cath_prot_pct,        cls: "bio" },
              { lbl: "% Seq Pos",          v: L.seq_pos_pct,          cls: "bio" },
            ];
            return (
              <button key={L.layer} className="layer-card" onClick={() => onPickLayer(L.layer)}>
                <div>
                  <div className="layer-num">
                    <span className="pre">Layer</span>{String(L.layer).padStart(2, "0")}
                  </div>
                </div>
                <div className="bar-stack">
                  {rows.map((r) => (
                    <div key={r.lbl} className="bar-row">
                      <span className="lbl">{r.lbl}</span>
                      <span className="v">{(r.v ?? 0).toFixed(2)}<span style={{fontSize:'10px',color:'var(--ink-3)',marginLeft:2}}>%</span></span>
                      <span className="bar-track"><span className={"bar-fill " + r.cls} style={{ width: `${r.v ?? 0}%` }}></span></span>
                    </div>
                  ))}
                </div>
                <div className="open-cta">
                  <span>{(L.novel_geom_count || 0).toLocaleString()} novel geom</span>
                  <span className="arrow">Open →</span>
                </div>
              </button>
            );
          })}
        </div>
      </section>

      {/* FEATURED EXAMPLES */}
      <section className="container section">
        <div className="section-head">
          <div>
            <span className="eyebrow">Step 2 — or jump straight in</span>
            <h3>Features described in the paper</h3>
          </div>
          <p className="desc">
            Every feature called out by ID in the paper, opened directly here.
            Includes the geometry-primary case, the ablation-causal feature,
            the four HSP90-like ATPase variants, the metagenomic case, and the
            schematic example.
          </p>
        </div>
        {featured.loading
          ? <Loading what="featured features" />
          : featured.error
            ? <ErrorBox err={featured.error} />
            : (
              <div className="examples-grid">
                {(featured.data || []).map((f) => (
                  <button key={`${f.layer}-${f.feature_id}`} className="example"
                          onClick={() => onPickFeature(f.layer, f.feature_id)}>
                    <div className="strip">
                      {/* Visualise the feature kind via the seq strip's coloration */}
                      {Array.from({ length: 24 }).map((_, i) => {
                        const dist = Math.abs(i - 12) / 12;
                        const a = Math.max(0, 1 - dist * 1.4);
                        const c = f.kind === "geom" ? "var(--geom)"
                                : f.kind === "bio"  ? "var(--bio)"
                                : "var(--both)";
                        return <span key={i} className="aa"
                          style={{ background: `color-mix(in oklch, ${c} ${Math.round(a * 60)}%, transparent)` }}>
                          {/* placeholder dot */}·
                        </span>;
                      })}
                    </div>
                    <div className="body">
                      <div className="head-row">
                        <span className="fid">L{f.layer}/f{f.feature_id}</span>
                        <Pill kind={f.kind === "geom" ? "geom" : f.kind === "bio" ? "bio" : "both"}>
                          {f.kind === "geom" ? "Geometry only" : f.kind === "bio" ? "Database only" : "Both"}
                        </Pill>
                      </div>
                      <div className="ttl">{f.title}</div>
                      <div className="desc">{f.desc}</div>
                      <div className="stats">
                        <span>geom <strong>{fmt(f.geom_score, 2)}</strong></span>
                        <span>best bio <strong>{fmt(f.bio_best, 2)}</strong></span>
                        <span>q<sub>geom</sub> <strong>{fmtQ(f.q_geom)}</strong></span>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            )}
      </section>
    </React.Fragment>
  );
}

window.Landing = Landing;
