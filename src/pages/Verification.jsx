import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Check, X, HelpCircle, ArrowRight } from 'lucide-react';

const Verification = () => {
  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchSamples();
  }, []);

  const fetchSamples = async () => {
    setLoading(true);
    try {
      const res = await axios.get('/api/samples?limit=10');
      setSamples(res.data);
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  const submitVerification = async (label) => {
    if (samples.length === 0) return;
    
    const current = samples[0];
    try {
      await axios.post('/api/verify', {
        sample_id: current.sample_id,
        label_faithfulness: label
      });
      // Remove verified sample from list
      setSamples(prev => prev.slice(1));
      
      // Fetch more if we run out
      if (samples.length <= 2) {
        fetchSamples();
      }
    } catch (err) {
      alert("Failed to submit verification: " + err.message);
    }
  };

  if (loading && samples.length === 0) return <div style={{padding: 40}}>Loading samples...</div>;
  if (error) return <div style={{padding: 40, color: 'red'}}>Error: {error}</div>;
  
  if (samples.length === 0) {
    return (
      <div className="promo-card" style={{background: 'var(--accent-green)', color: 'var(--text-primary)'}}>
        <h2 className="promo-title" style={{maxWidth: '100%'}}>All Caught Up!</h2>
        <p className="promo-text" style={{maxWidth: '100%', color: 'var(--text-secondary)'}}>
          There are no more unverified samples in the dataset.
        </p>
        <button className="btn-primary" onClick={fetchSamples}>Refresh</button>
      </div>
    );
  }

  const current = samples[0];

  return (
    <div style={{maxWidth: '800px', margin: '0 auto'}}>
      
      <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: 20}}>
        <div style={{background: '#E2E8F0', padding: '4px 12px', borderRadius: 16, fontSize: 12, fontWeight: 600}}>
          Sample ID: {current.sample_id}
        </div>
        {current.label === 1 && (
          <div style={{background: '#FED7D7', color: '#C53030', padding: '4px 12px', borderRadius: 16, fontSize: 12, fontWeight: 600}}>
            Weak Label: Hallucinated ({current.injection_strategy})
          </div>
        )}
        {current.label === 0 && (
          <div style={{background: '#C6F6D5', color: '#276749', padding: '4px 12px', borderRadius: 16, fontSize: 12, fontWeight: 600}}>
            Weak Label: Faithful
          </div>
        )}
      </div>

      <div style={{background: 'white', border: '1px solid var(--border-color)', borderRadius: 24, padding: 32, marginBottom: 24, boxShadow: '0 4px 12px rgba(0,0,0,0.03)'}}>
        
        <div style={{marginBottom: 24}}>
          <h3 style={{fontSize: 14, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8}}>Context</h3>
          <div style={{background: '#F7FAFC', padding: 16, borderRadius: 12, fontSize: 15, lineHeight: 1.6, maxHeight: '200px', overflowY: 'auto'}}>
            {current.context}
          </div>
        </div>

        <div style={{marginBottom: 24}}>
          <h3 style={{fontSize: 14, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8}}>Question</h3>
          <div style={{fontSize: 18, fontWeight: 600}}>
            {current.question}
          </div>
        </div>
        
        <div style={{marginBottom: 32}}>
          <h3 style={{fontSize: 14, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8}}>Reference Answer (Ground Truth)</h3>
          <div style={{color: '#2B6CB0', fontWeight: 500, fontSize: 16}}>
            {current.reference_answer}
          </div>
        </div>

        <div style={{borderTop: '1px solid var(--border-color)', paddingTop: 24}}>
          <h3 style={{fontSize: 14, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12}}>Model Generated Answer</h3>
          <div style={{fontSize: 20, fontWeight: 400, padding: 16, borderLeft: '4px solid var(--accent-blue)', background: '#F8FAFC'}}>
            {current.generated_answer}
          </div>
        </div>
      </div>

      <h3 style={{textAlign: 'center', marginBottom: 16, color: 'var(--text-secondary)'}}>Is the generated answer faithful to the context?</h3>
      
      <div style={{display: 'flex', gap: 16, justifyContent: 'center'}}>
        <button 
          onClick={() => submitVerification('faithful')}
          style={{flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, padding: 24, borderRadius: 16, border: '2px solid #C6F6D5', background: 'white', cursor: 'pointer', transition: 'var(--transition)'}}>
          <div style={{width: 48, height: 48, borderRadius: '50%', background: '#C6F6D5', color: '#276749', display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
            <Check size={24} />
          </div>
          <span style={{fontWeight: 600, color: '#276749'}}>Faithful</span>
        </button>

        <button 
          onClick={() => submitVerification('hallucinated')}
          style={{flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, padding: 24, borderRadius: 16, border: '2px solid #FED7D7', background: 'white', cursor: 'pointer', transition: 'var(--transition)'}}>
          <div style={{width: 48, height: 48, borderRadius: '50%', background: '#FED7D7', color: '#C53030', display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
            <X size={24} />
          </div>
          <span style={{fontWeight: 600, color: '#C53030'}}>Hallucinated</span>
        </button>

        <button 
          onClick={() => submitVerification('uncertain')}
          style={{flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, padding: 24, borderRadius: 16, border: '2px solid #E2E8F0', background: 'white', cursor: 'pointer', transition: 'var(--transition)'}}>
          <div style={{width: 48, height: 48, borderRadius: '50%', background: '#E2E8F0', color: '#4A5568', display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
            <HelpCircle size={24} />
          </div>
          <span style={{fontWeight: 600, color: '#4A5568'}}>Uncertain</span>
        </button>
      </div>

    </div>
  );
};

export default Verification;
