import React from 'react';
import { Server } from 'lucide-react';
import useDashboardData, { LANG_NAMES, STRATEGY_NAMES, fmt, card, th, td, Bar, EmptyState } from '../useDashboardData';

const Generation = () => {
  const { data, error, loading } = useDashboardData();
  if (loading) return <div style={{ padding: 40 }}>Loading Phase 2 outputs...</div>;
  if (error) return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;
  const p2 = data.phase2;
  if (!p2) return <EmptyState phase="Phase 2" />;

  const maxStrategy = Math.max(...Object.values(p2.strategies));

  return (
    <div style={{ maxWidth: 1000 }}>
      <div style={{ ...card, marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 20 }}>
          <div style={{ width: 48, height: 48, borderRadius: 12, background: 'var(--accent-blue)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Server size={24} />
          </div>
          <div>
            <h2 style={{ fontSize: 20, fontWeight: 600, margin: 0 }}>Phase 2: Faithful and Hallucinated Answer Generation</h2>
            <p style={{ margin: 0, color: 'var(--text-secondary)', marginTop: 4 }}>
              {p2.model ? `Groq · ${p2.model}` : 'Groq API'} · {fmt(p2.samples)} source samples → {fmt(p2.rows)} labeled rows
            </p>
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          {[
            ['Samples', fmt(p2.samples)],
            ['Rows (label 0 + 1)', fmt(p2.rows)],
            ['Quality-guard flags', fmt(p2.flagged_total)],
            ['Temperature (faithful / halluc.)', p2.temperatures || '0.1 / 0.7'],
          ].map(([k, v]) => (
            <div key={k} style={{ padding: 16, border: '1px solid var(--border-color)', borderRadius: 12 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: 6 }}>{k}</div>
              <div style={{ fontWeight: 600, fontSize: 18 }}>{v}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.3fr 1fr', gap: 24, marginBottom: 24 }}>
        <div style={card}>
          <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 12 }}>Rows by language</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr><th style={th}>Language</th><th style={th}>Faithful</th><th style={th}>Hallucinated</th><th style={th}>Flagged</th></tr></thead>
            <tbody>
              {['hi', 'mr', 'ta'].map(lc => (
                <tr key={lc}>
                  <td style={td}>{LANG_NAMES[lc]}</td>
                  <td style={td}>{fmt(p2.by_language[lc].faithful)}</td>
                  <td style={td}>{fmt(p2.by_language[lc].hallucinated)}</td>
                  <td style={td}>{fmt(p2.by_language[lc].flagged)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={card}>
          <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 16 }}>Injection strategies used</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {Object.keys(STRATEGY_NAMES).map(s => (
              <div key={s}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 14, marginBottom: 6 }}>
                  <span>{STRATEGY_NAMES[s]}</span><b>{fmt(p2.strategies[s] || 0)}</b>
                </div>
                <Bar value={p2.strategies[s] || 0} max={maxStrategy} />
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={card}>
        <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 12 }}>Example pairs</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead><tr><th style={th}>Lang</th><th style={th}>Question</th><th style={th}>Reference</th><th style={th}>Faithful</th><th style={th}>Hallucinated</th><th style={th}>Strategy</th></tr></thead>
          <tbody>
            {p2.examples.slice(0, 9).map((e, i) => (
              <tr key={i}>
                <td style={td}>{e.language_code}</td>
                <td style={{ ...td, maxWidth: 260 }}>{e.question}</td>
                <td style={{ ...td, color: '#2B6CB0' }}>{e.reference_answer}</td>
                <td style={{ ...td, color: '#276749' }}>{e.faithful_answer}</td>
                <td style={{ ...td, color: '#C53030' }}>{e.hallucinated_answer}</td>
                <td style={{ ...td, fontSize: 12 }}>{STRATEGY_NAMES[e.injection_strategy]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Generation;
