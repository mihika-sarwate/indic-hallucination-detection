import React from 'react';
import { useNavigate } from 'react-router-dom';
import { FileText, Layers, CheckCircle2, BarChart } from 'lucide-react';
import useDashboardData, { LANG_NAMES, fmt, pct, EmptyState } from '../useDashboardData';

const Dashboard = () => {
  const navigate = useNavigate();
  const { data, error, loading } = useDashboardData();

  if (loading) return <div style={{ padding: 40 }}>Loading pipeline outputs...</div>;
  if (error) return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;
  if (!data.phase1) return <EmptyState phase="Phase 1" />;

  const p1 = data.phase1;
  const p2 = data.phase2;
  const p3 = data.phase3;
  const p45 = data.phase45;
  const bestModel = p45 && Object.entries(p45.models)
    .sort((a, b) => b[1].summary.overall.f1.mean - a[1].summary.overall.f1.mean)[0];

  const stats = [
    { label: 'Clean QA examples (Phase 1)', value: fmt(p1.valid_total), icon: <FileText size={24} />, color: 'blue' },
    { label: 'Faithful + hallucinated rows (Phase 2)', value: p2 ? fmt(p2.rows) : '—', icon: <Layers size={24} />, color: 'purple' },
    { label: 'Labels checked (Phase 3)', value: p3 ? fmt(p3.sample_size) : '—', icon: <CheckCircle2 size={24} />, color: 'green' },
    { label: bestModel ? `Best test F1 (${bestModel[0]})` : 'Best test F1 (Phase 5)', value: bestModel ? bestModel[1].summary.overall.f1.mean.toFixed(3) : '—', icon: <BarChart size={24} />, color: 'yellow' },
  ];

  return (
    <>
      <div className="dashboard-grid">
        {stats.map((stat, idx) => (
          <div key={idx} className={`stat-card ${stat.color}`}>
            <div>
              <div className="stat-value">{stat.value}</div>
              <div className="stat-label">{stat.label}</div>
            </div>
            <div className="stat-icon-wrapper">{stat.icon}</div>
          </div>
        ))}
      </div>

      <div className="section-header">
        <h2 className="section-title">Language Breakdown</h2>
      </div>

      <div className="list-container">
        {['hi', 'mr', 'ta'].map(lc => {
          const l = p1.by_language[lc];
          const g = p2 && p2.by_language[lc];
          const f1 = bestModel && bestModel[1].summary.by_language[lc].f1.mean;
          return (
            <div key={lc} className="list-item">
              <div className="item-icon">{LANG_NAMES[lc].charAt(0)}</div>
              <div>
                <div style={{ fontWeight: 600 }}>{LANG_NAMES[lc]} ({lc})</div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                  {fmt(l.splits.train)} train · {fmt(l.splits.validation)} val · {fmt(l.splits.test)} test
                </div>
              </div>
              <div style={{ fontWeight: 500, color: 'var(--text-secondary)' }}>{fmt(l.original)} raw → {fmt(l.valid)} valid</div>
              <div style={{ color: '#38A169', fontWeight: 500 }}>{g ? `${fmt(g.faithful + g.hallucinated)} Phase 2 rows` : 'Phase 2 pending'}</div>
              <div style={{ fontWeight: 500 }}>{f1 !== undefined && f1 !== null ? `F1 ${f1.toFixed(3)}` : '—'}</div>
              <div>
                <button className="btn-primary" style={{ padding: '8px 16px' }} onClick={() => navigate('/evaluation')}>Results</button>
              </div>
            </div>
          );
        })}
      </div>

      <div className="section-header" style={{ marginTop: 24 }}>
        <h2 className="section-title">Phase 1 removals</h2>
      </div>
      <div className="list-container">
        <div className="list-item" style={{ gridTemplateColumns: '1fr 1fr 1fr 1fr' }}>
          <div><b>{fmt(p1.original_total)}</b> raw rows</div>
          <div><b>{fmt(p1.removed_total)}</b> removed ({pct(p1.removed_total / p1.original_total)})</div>
          <div><b>{fmt(p1.removal_reasons.missing_reference_answer)}</b> missing answer</div>
          <div><b>{fmt(p1.removal_reasons.exact_duplicates + p1.removal_reasons.context_question_duplicates)}</b> duplicates</div>
        </div>
      </div>
    </>
  );
};

export default Dashboard;
