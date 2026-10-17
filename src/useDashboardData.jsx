import { useEffect, useState } from 'react';
import axios from 'axios';

// Real pipeline outputs, collected by src/build_dashboard_data.py.
let cache = null;

export default function useDashboardData() {
  const [data, setData] = useState(cache);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (cache) return;
    axios.get('./dashboard_data.json')
      .then(res => { cache = res.data; setData(res.data); })
      .catch(err => setError(err.message));
  }, []);

  return { data, error, loading: !data && !error };
}

export const LANG_NAMES = { hi: 'Hindi', mr: 'Marathi', ta: 'Tamil' };

export const STRATEGY_NAMES = {
  entity_substitution: 'Entity substitution',
  numeric_date_corruption: 'Numeric / date corruption',
  unsupported_claim_addition: 'Unsupported claim addition',
  contradicted_fact: 'Contradicted fact',
};

export const fmt = n => (n === undefined || n === null ? '—' : Number(n).toLocaleString('en-US'));
export const pct = (v, digits = 1) => (v === undefined || v === null ? '—' : `${(v * 100).toFixed(digits)}%`);

export const card = { background: 'white', padding: 24, borderRadius: 16, border: '1px solid var(--border-color)' };
export const th = { padding: '12px 16px', fontWeight: 600, color: 'var(--text-secondary)', fontSize: 12, textTransform: 'uppercase', textAlign: 'left' };
export const td = { padding: '12px 16px', fontSize: 14, borderTop: '1px solid var(--border-color)' };

export function EmptyState({ phase }) {
  return (
    <div style={{ ...card, color: 'var(--text-secondary)' }}>
      {phase} has not been run yet, so there is nothing to show. Run the pipeline and rebuild dashboard_data.json.
    </div>
  );
}

export function Bar({ value, max = 1, color = '#3182CE' }) {
  return (
    <div style={{ width: '100%', height: 8, background: '#E2E8F0', borderRadius: 4, overflow: 'hidden' }}>
      <div style={{ width: `${Math.min(100, (value / max) * 100)}%`, height: '100%', background: color }} />
    </div>
  );
}
