import React, { useState } from 'react';
import { Play, Loader2, CheckCircle2, Server } from 'lucide-react';

const Generation = () => {
  const [status, setStatus] = useState('idle'); // idle, generating, complete
  const [progress, setProgress] = useState(0);

  const startGeneration = () => {
    setStatus('generating');
    setProgress(0);
    
    // Simulate generation progress
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setStatus('complete');
          return 100;
        }
        return prev + Math.floor(Math.random() * 15);
      });
    }, 1000);
  };

  return (
    <div style={{ maxWidth: 800 }}>
      <div style={{ background: 'white', padding: 32, borderRadius: 16, border: '1px solid var(--border-color)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 24 }}>
          <div style={{ width: 48, height: 48, borderRadius: 12, background: 'var(--primary-light)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--primary-color)' }}>
            <Server size={24} />
          </div>
          <div>
            <h2 style={{ fontSize: 20, fontWeight: 600, margin: 0, color: 'var(--text-primary)' }}>LLM Hallucination Pipeline</h2>
            <p style={{ margin: 0, color: 'var(--text-secondary)', marginTop: 4 }}>Generate faithful and hallucinated answers for your dataset using Groq.</p>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 32 }}>
          <div style={{ padding: 16, border: '1px solid var(--border-color)', borderRadius: 12 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: 8 }}>Input Source</div>
            <div style={{ fontWeight: 500 }}>data/phase1/cleaned_dataset.csv</div>
          </div>
          <div style={{ padding: 16, border: '1px solid var(--border-color)', borderRadius: 12 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: 8 }}>Languages</div>
            <div style={{ fontWeight: 500 }}>Hindi, Marathi, Tamil</div>
          </div>
        </div>

        {status === 'idle' && (
          <button onClick={startGeneration} className="btn-primary" style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, padding: 16, fontSize: 16 }}>
            <Play size={20} />
            Start Generation Pipeline
          </button>
        )}

        {status === 'generating' && (
          <div style={{ padding: 24, background: '#F7FAFC', borderRadius: 12, border: '1px solid #E2E8F0' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 500, color: '#2B6CB0' }}>
                <Loader2 size={18} className="spin" />
                Generating responses via Groq API...
              </div>
              <div style={{ fontWeight: 600, color: '#2D3748' }}>{progress}%</div>
            </div>
            <div style={{ width: '100%', height: 8, background: '#E2E8F0', borderRadius: 4, overflow: 'hidden' }}>
              <div style={{ width: `${progress}%`, height: '100%', background: '#3182CE', transition: 'width 0.3s ease' }} />
            </div>
          </div>
        )}

        {status === 'complete' && (
          <div style={{ padding: 24, background: '#F0FFF4', borderRadius: 12, border: '1px solid #C6F6D5', display: 'flex', alignItems: 'center', gap: 12 }}>
            <CheckCircle2 size={24} color="#38A169" />
            <div>
              <div style={{ fontWeight: 600, color: '#276749' }}>Generation Complete!</div>
              <div style={{ fontSize: 14, color: '#2F855A', marginTop: 4 }}>Dataset successfully saved to data/phase2/phase2_dataset.csv</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Generation;
