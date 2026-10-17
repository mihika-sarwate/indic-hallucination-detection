import React from 'react';
import { BarChart, CheckCircle2, AlertTriangle, Globe } from 'lucide-react';

const Evaluation = () => {
  const metrics = [
    { lang: 'Hindi (hi)', f1: 0.89, precision: 0.91, recall: 0.88 },
    { lang: 'Marathi (mr)', f1: 0.86, precision: 0.88, recall: 0.85 },
    { lang: 'Tamil (ta)', f1: 0.82, precision: 0.84, recall: 0.81 },
  ];

  return (
    <div style={{ maxWidth: 900 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 24 }}>
        <div style={{ width: 48, height: 48, borderRadius: 12, background: 'var(--primary-light)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--primary-color)' }}>
          <BarChart size={24} />
        </div>
        <div>
          <h2 style={{ fontSize: 24, fontWeight: 600, margin: 0, color: 'var(--text-primary)' }}>Phase 5: Evaluation & Comparison</h2>
          <p style={{ margin: 0, color: 'var(--text-secondary)', marginTop: 4 }}>Cross-lingual hallucination detection performance metrics.</p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 24 }}>
        <div style={{ background: 'white', padding: 24, borderRadius: 16, border: '1px solid var(--border-color)' }}>
          <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 16, color: 'var(--text-primary)' }}>Performance by Language</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {metrics.map((m, i) => (
              <div key={i}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, fontSize: 14 }}>
                  <span style={{ fontWeight: 500, display: 'flex', alignItems: 'center', gap: 6 }}><Globe size={14} /> {m.lang}</span>
                  <span style={{ fontWeight: 600, color: 'var(--primary-color)' }}>F1: {m.f1}</span>
                </div>
                <div style={{ width: '100%', height: 8, background: '#E2E8F0', borderRadius: 4, overflow: 'hidden' }}>
                  <div style={{ width: `${m.f1 * 100}%`, height: '100%', background: '#3182CE' }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div style={{ background: 'white', padding: 24, borderRadius: 16, border: '1px solid var(--border-color)' }}>
          <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 16, color: 'var(--text-primary)' }}>Error Analysis (Hardest to Catch)</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div style={{ padding: 12, background: '#FFF5F5', borderRadius: 8, border: '1px solid #FED7D7' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 600, color: '#C53030', marginBottom: 4 }}>
                <AlertTriangle size={16} /> Unsupported Claims
              </div>
              <div style={{ fontSize: 13, color: '#9B2C2C' }}>Model frequently misses hallucinations where completely novel, plausible-sounding facts are invented.</div>
            </div>
            <div style={{ padding: 12, background: '#F0FFF4', borderRadius: 8, border: '1px solid #C6F6D5' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 600, color: '#276749', marginBottom: 4 }}>
                <CheckCircle2 size={16} /> Entity Substitution
              </div>
              <div style={{ fontSize: 13, color: '#2F855A' }}>Model successfully catches 94% of cases where names, dates, or numbers are swapped.</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Evaluation;
