import React from 'react';
import { BarChart, Globe } from 'lucide-react';
import useDashboardData, { LANG_NAMES, STRATEGY_NAMES, card, th, td, Bar, EmptyState } from '../useDashboardData';

const f3 = v => (v === undefined || v === null ? '—' : v.toFixed(3));

const Evaluation = () => {
  const { data, error, loading } = useDashboardData();
  if (loading) return <div style={{ padding: 40 }}>Loading evaluation results...</div>;
  if (error) return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;
  const p45 = data.phase45;
  if (!p45) return <EmptyState phase="Phase 5" />;

  const rows = [
    ...Object.entries(p45.models).map(([name, m]) => ({ name, overall: m.summary.overall, byLang: m.summary.by_language, seeds: true })),
    ...Object.entries(p45.baselines).map(([name, b]) => ({ name, overall: b.overall, byLang: b.by_language, seeds: false })),
  ];
  const val = (x, k) => (x[k] && typeof x[k] === 'object' ? x[k].mean : x[k]);
  const cross = p45.cross_lingual;

  return (
    <div style={{ maxWidth: 1000 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 24 }}>
        <div style={{ width: 48, height: 48, borderRadius: 12, background: 'var(--accent-blue)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <BarChart size={24} />
        </div>
        <div>
          <h2 style={{ fontSize: 24, fontWeight: 600, margin: 0 }}>Phase 5: Evaluation & Comparison</h2>
          <p style={{ margin: 0, color: 'var(--text-secondary)', marginTop: 4 }}>Test split · hallucinated = positive class · transformer scores are means over {p45.config.seeds.length} seeds</p>
        </div>
      </div>

      <div style={{ ...card, marginBottom: 24 }}>
        <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 12 }}>Overall test results</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead><tr><th style={th}>Model</th><th style={th}>Accuracy</th><th style={th}>Precision</th><th style={th}>Recall</th><th style={th}>F1</th><th style={th}>Macro-F1</th></tr></thead>
          <tbody>
            {rows.map(r => (
              <tr key={r.name}>
                <td style={{ ...td, fontWeight: 600 }}>{r.name}</td>
                {['accuracy', 'precision', 'recall', 'f1', 'macro_f1'].map(k => (
                  <td key={k} style={td}>{f3(val(r.overall, k))}{r.seeds ? <span style={{ color: 'var(--text-secondary)', fontSize: 12 }}> ± {r.overall[k].std.toFixed(3)}</span> : null}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 24 }}>
        <div style={card}>
          <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 16 }}>F1 by language</h3>
          {['hi', 'mr', 'ta'].map(lc => (
            <div key={lc} style={{ marginBottom: 16 }}>
              <div style={{ fontWeight: 500, display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, fontSize: 14 }}><Globe size={14} /> {LANG_NAMES[lc]}</div>
              {rows.map(r => (
                <div key={r.name} style={{ display: 'grid', gridTemplateColumns: '110px 1fr 48px', alignItems: 'center', gap: 8, fontSize: 13, marginBottom: 4 }}>
                  <span>{r.name}</span><Bar value={val(r.byLang[lc], 'f1')} color={r.seeds ? '#3182CE' : '#A0AEC0'} /><b>{f3(val(r.byLang[lc], 'f1'))}</b>
                </div>
              ))}
            </div>
          ))}
        </div>
        <div style={card}>
          <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 16 }}>Recall by hallucination type</h3>
          {Object.keys(STRATEGY_NAMES).map(s => (
            <div key={s} style={{ marginBottom: 16 }}>
              <div style={{ fontWeight: 500, marginBottom: 8, fontSize: 14 }}>{STRATEGY_NAMES[s]}</div>
              {Object.entries(p45.models).map(([name, m]) => (
                <div key={name} style={{ display: 'grid', gridTemplateColumns: '110px 1fr 48px', alignItems: 'center', gap: 8, fontSize: 13, marginBottom: 4 }}>
                  <span>{name}</span><Bar value={m.summary.by_strategy_recall[s].mean} /><b>{f3(m.summary.by_strategy_recall[s].mean)}</b>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>

      {cross && (
        <div style={card}>
          <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 12 }}>Cross-lingual transfer ({cross.model}, F1): rows = training language, columns = test language</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr><th style={th}>Train ↓ / Test →</th>{['hi', 'mr', 'ta'].map(lc => <th key={lc} style={th}>{LANG_NAMES[lc]}</th>)}</tr></thead>
            <tbody>
              {['hi', 'mr', 'ta'].map(src => (
                <tr key={src}>
                  <td style={{ ...td, fontWeight: 600 }}>{LANG_NAMES[src]}</td>
                  {['hi', 'mr', 'ta'].map(tgt => (
                    <td key={tgt} style={{ ...td, background: src === tgt ? '#EBF8FF' : 'transparent' }}>{f3(cross.f1[src][tgt])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default Evaluation;
