import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Dataset = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [lang, setLang] = useState('All');

  useEffect(() => {
    fetchData();
  }, [lang]);

  const fetchData = async () => {
    setLoading(true);
    try {
      const url = lang === 'All' ? '/api/dataset' : `/api/dataset?lang=${lang}`;
      const res = await axios.get(url);
      setData(res.data);
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  return (
    <div>
      <div style={{marginBottom: 20, display: 'flex', alignItems: 'center', gap: 10}}>
        <label style={{fontWeight: 600, color: 'var(--text-primary)'}}>Filter by Language:</label>
        <select 
          value={lang} 
          onChange={(e) => setLang(e.target.value)}
          style={{padding: '8px 12px', borderRadius: 8, border: '1px solid var(--border-color)', outline: 'none'}}
        >
          <option value="All">All Languages</option>
          <option value="Hindi">Hindi</option>
          <option value="Marathi">Marathi</option>
          <option value="Tamil">Tamil</option>
        </select>
      </div>
      
      {loading ? <div style={{padding: 40}}>Loading dataset...</div> : (
        <div style={{background: 'white', borderRadius: 16, border: '1px solid var(--border-color)', overflow: 'hidden'}}>
          <div style={{overflowX: 'auto'}}>
            <table style={{width: '100%', borderCollapse: 'collapse', textAlign: 'left'}}>
              <thead>
                <tr style={{background: '#F7FAFC', borderBottom: '1px solid var(--border-color)'}}>
                  <th style={{padding: '16px', fontWeight: 600, color: 'var(--text-secondary)', fontSize: 12, textTransform: 'uppercase'}}>ID</th>
                  <th style={{padding: '16px', fontWeight: 600, color: 'var(--text-secondary)', fontSize: 12, textTransform: 'uppercase'}}>Language</th>
                  <th style={{padding: '16px', fontWeight: 600, color: 'var(--text-secondary)', fontSize: 12, textTransform: 'uppercase'}}>Question</th>
                  <th style={{padding: '16px', fontWeight: 600, color: 'var(--text-secondary)', fontSize: 12, textTransform: 'uppercase'}}>Reference</th>
                  <th style={{padding: '16px', fontWeight: 600, color: 'var(--text-secondary)', fontSize: 12, textTransform: 'uppercase'}}>Generated</th>
                </tr>
              </thead>
              <tbody>
                {data.map((row, idx) => (
                  <tr key={idx} style={{borderBottom: '1px solid var(--border-color)'}}>
                    <td style={{padding: '16px', fontSize: 13, maxWidth: 100, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'}} title={row.sample_id}>{row.sample_id}</td>
                    <td style={{padding: '16px', fontSize: 14}}>{row.language}</td>
                    <td style={{padding: '16px', fontSize: 14, maxWidth: 200}}>{row.question}</td>
                    <td style={{padding: '16px', fontSize: 14, maxWidth: 200, color: '#2B6CB0'}}>{row.reference_answer}</td>
                    <td style={{padding: '16px', fontSize: 14, maxWidth: 200}}>{row.generated_answer}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dataset;
