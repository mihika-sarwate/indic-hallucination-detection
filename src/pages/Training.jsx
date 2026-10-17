import React from 'react';
import { Cpu } from 'lucide-react';
import useDashboardData, { fmt, card, th, td, EmptyState } from '../useDashboardData';

const Training = () => {
  const { data, error, loading } = useDashboardData();
  if (loading) return <div style={{ padding: 40 }}>Loading training logs...</div>;
  if (error) return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;
  const p45 = data.phase45;
  if (!p45) return <EmptyState phase="Phase 4" />;
  const cfg = p45.config;

  return (
    <div style={{ maxWidth: 1000 }}>
      <div style={{ ...card, marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 20 }}>
          <div style={{ width: 48, height: 48, borderRadius: 12, background: 'var(--accent-blue)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Cpu size={24} />
          </div>
          <div>
            <h2 style={{ fontSize: 20, fontWeight: 600, margin: 0 }}>Phase 4: Classifier Training</h2>
            <p style={{ margin: 0, color: 'var(--text-secondary)', marginTop: 4 }}>
              Sentence-pair fine-tuning (question + context, candidate answer) · {cfg.epochs} epochs · max length {cfg.max_len} · seeds {cfg.seeds.join(', ')}
            </p>
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          {[
            ['Train rows', fmt(p45.data.rows_used_by_split.train)],
            ['Validation rows', fmt(p45.data.rows_used_by_split.validation)],
            ['Test rows', fmt(p45.data.rows_used_by_split.test)],
            ['Flagged rows excluded', fmt(p45.data.rows_flagged)],
          ].map(([k, v]) => (
            <div key={k} style={{ padding: 16, border: '1px solid var(--border-color)', borderRadius: 12 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: 6 }}>{k}</div>
              <div style={{ fontWeight: 600, fontSize: 18 }}>{v}</div>
            </div>
          ))}
        </div>
      </div>

      {Object.entries(p45.models).map(([name, m]) => (
        <div key={name} style={{ ...card, marginBottom: 24 }}>
          <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 4 }}>{name} <span style={{ fontWeight: 400, color: 'var(--text-secondary)' }}>({m.hf_name})</span></h3>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 0 }}>
            Learning rate {Number(cfg.models[name].lr).toExponential()} · AdamW · best epochs per seed: {m.best_epochs.join(', ')} · training time per run: {m.train_seconds.map(s => `${Math.round(s)} s`).join(', ')}
          </p>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr><th style={th}>Epoch</th><th style={th}>Train loss</th><th style={th}>Val accuracy</th><th style={th}>Val F1</th><th style={th}>Val macro-F1</th></tr></thead>
            <tbody>
              {m.history.map(h => (
                <tr key={h.epoch} style={h.epoch === m.best_epochs[0] ? { background: '#F0FFF4' } : {}}>
                  <td style={td}>{h.epoch}{h.epoch === m.best_epochs[0] ? ' (selected)' : ''}</td>
                  <td style={td}>{h.train_loss.toFixed(4)}</td>
                  <td style={td}>{h.val_accuracy.toFixed(3)}</td>
                  <td style={td}>{h.val_f1.toFixed(3)}</td>
                  <td style={td}>{h.val_macro_f1.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 0 }}>Epoch log shown for seed {m.seeds[0]}.</p>
        </div>
      ))}
    </div>
  );
};

export default Training;
