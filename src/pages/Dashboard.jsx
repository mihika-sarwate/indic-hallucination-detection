import React from 'react';
import { FileText, CheckCircle2, AlertTriangle, Layers } from 'lucide-react';

const Dashboard = () => {
  const stats = [
    { label: 'Total Samples', value: '3,437', icon: <FileText size={24} />, color: 'blue' },
    { label: 'Generated Pairs', value: '6,874', icon: <Layers size={24} />, color: 'purple' },
    { label: 'Faithful Verified', value: '450', icon: <CheckCircle2 size={24} />, color: 'green' },
    { label: 'Hallucinated Verified', value: '423', icon: <AlertTriangle size={24} />, color: 'yellow' },
  ];

  const languages = [
    { name: 'Hindi (hi)', samples: '1,052', split: 'Test Set', verified: '280' },
    { name: 'Marathi (mr)', samples: '1,108', split: 'Test Set', verified: '293' },
    { name: 'Tamil (ta)', samples: '1,277', split: 'Test Set', verified: '300' },
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
            <div className="stat-icon-wrapper">
              {stat.icon}
            </div>
          </div>
        ))}
      </div>

      <div className="section-header">
        <h2 className="section-title">Language Breakdown</h2>
      </div>

      <div className="list-container">
        {languages.map((lang, idx) => (
          <div key={idx} className="list-item">
            <div className="item-icon">
              {lang.name.charAt(0)}
            </div>
            <div>
              <div style={{fontWeight: 600}}>{lang.name}</div>
              <div style={{fontSize: 12, color: 'var(--text-secondary)'}}>{lang.split}</div>
            </div>
            <div style={{fontWeight: 500, color: 'var(--text-secondary)'}}>{lang.samples} Total</div>
            <div style={{color: '#38A169', fontWeight: 500}}>+{lang.verified} Verified</div>
            <div>
               <div style={{width: '100px', height: '6px', background: '#E2E8F0', borderRadius: '3px'}}>
                  <div style={{width: `${(parseInt(lang.verified)/parseInt(lang.samples.replace(',','')))*100}%`, height: '100%', background: '#3182CE', borderRadius: '3px'}}></div>
               </div>
            </div>
            <div>
              <button className="btn-primary" style={{padding: '8px 16px'}}>Review</button>
            </div>
          </div>
        ))}
      </div>
    </>
  );
};

export default Dashboard;
