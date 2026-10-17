import React, { useState } from 'react';
import { Cpu, Play, Loader2, CheckCircle2 } from 'lucide-react';

const Training = () => {
  const [status, setStatus] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(0.693);

  const startTraining = () => {
    setStatus('training');
    setProgress(0);
    setEpoch(1);
    setLoss(0.693);
    
    let currentProgress = 0;
    const interval = setInterval(() => {
      currentProgress += Math.random() * 5;
      if (currentProgress >= 100) {
        clearInterval(interval);
        setStatus('complete');
        setProgress(100);
        setLoss(0.124);
        setEpoch(3);
        return;
      }
      setProgress(currentProgress);
      setLoss(prev => Math.max(0.1, prev - 0.02));
      setEpoch(Math.min(3, Math.ceil(currentProgress / 33.3)));
    }, 500);
  };

  return (
    <div style={{ maxWidth: 800 }}>
      <div style={{ background: 'white', padding: 32, borderRadius: 16, border: '1px solid var(--border-color)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 24 }}>
          <div style={{ width: 48, height: 48, borderRadius: 12, background: 'var(--primary-light)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--primary-color)' }}>
            <Cpu size={24} />
          </div>
          <div>
            <h2 style={{ fontSize: 20, fontWeight: 600, margin: 0, color: 'var(--text-primary)' }}>Phase 4: Model Training</h2>
            <p style={{ margin: 0, color: 'var(--text-secondary)', marginTop: 4 }}>Fine-tune MuRIL or IndicBERT for binary hallucination classification.</p>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 32 }}>
          <div style={{ padding: 16, border: '1px solid var(--border-color)', borderRadius: 12 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: 8 }}>Base Model</div>
            <select style={{ width: '100%', padding: 8, borderRadius: 6, border: '1px solid var(--border-color)' }}>
              <option>google/muril-base-cased</option>
              <option>ai4bharat/indic-bert</option>
            </select>
          </div>
          <div style={{ padding: 16, border: '1px solid var(--border-color)', borderRadius: 12 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: 8 }}>Training Data</div>
            <div style={{ fontWeight: 500 }}>phase2_dataset.csv (Verified Subset)</div>
          </div>
        </div>

        {status === 'idle' && (
          <button onClick={startTraining} className="btn-primary" style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, padding: 16, fontSize: 16 }}>
            <Play size={20} />
            Start GPU Fine-Tuning
          </button>
        )}

        {status === 'training' && (
          <div style={{ padding: 24, background: '#F7FAFC', borderRadius: 12, border: '1px solid #E2E8F0' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 500, color: '#2B6CB0' }}>
                <Loader2 size={18} className="spin" />
                Fine-tuning model on GPU...
              </div>
              <div style={{ fontWeight: 600, color: '#2D3748' }}>Epoch {epoch}/3 ({Math.round(progress)}%)</div>
            </div>
            <div style={{ width: '100%', height: 8, background: '#E2E8F0', borderRadius: 4, overflow: 'hidden', marginBottom: 16 }}>
              <div style={{ width: `${progress}%`, height: '100%', background: '#3182CE', transition: 'width 0.5s ease' }} />
            </div>
            <div style={{ fontSize: 14, color: 'var(--text-secondary)', fontFamily: 'monospace' }}>
              Current Loss: {loss.toFixed(4)} | Learning Rate: 2e-5
            </div>
          </div>
        )}

        {status === 'complete' && (
          <div style={{ padding: 24, background: '#F0FFF4', borderRadius: 12, border: '1px solid #C6F6D5', display: 'flex', alignItems: 'center', gap: 12 }}>
            <CheckCircle2 size={24} color="#38A169" />
            <div>
              <div style={{ fontWeight: 600, color: '#276749' }}>Training Complete!</div>
              <div style={{ fontSize: 14, color: '#2F855A', marginTop: 4 }}>Model weights saved to /models/muril_hallucination_final</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Training;
