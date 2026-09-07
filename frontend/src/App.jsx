import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import { LayoutDashboard, Database, Activity, CheckCircle, Search, Bell, Layers } from 'lucide-react';
import Dashboard from './pages/Dashboard';
import Verification from './pages/Verification';
import './index.css';

const Sidebar = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const navItems = [
    { id: '/', icon: <LayoutDashboard size={20} /> },
    { id: '/dataset', icon: <Database size={20} /> },
    { id: '/generation', icon: <Activity size={20} /> },
    { id: '/verification', icon: <CheckCircle size={20} /> },
  ];

  return (
    <div className="sidebar">
      <div className="logo-container">
        <Layers size={24} strokeWidth={2.5} />
      </div>
      <div className="nav-menu">
        {navItems.map(item => (
          <div 
            key={item.id}
            className={`nav-item ${location.pathname === item.id ? 'active' : ''}`}
            onClick={() => navigate(item.id)}
          >
            {item.icon}
          </div>
        ))}
      </div>
    </div>
  );
};

const TopBar = () => {
  const location = useLocation();
  const titleMap = {
    '/': 'Overview',
    '/dataset': 'Dataset Management',
    '/generation': 'Phase 2 Generation',
    '/verification': 'Human Verification',
  };
  
  const title = titleMap[location.pathname] || 'Dashboard';

  return (
    <div className="top-bar">
      <h1 className="page-title">{title}</h1>
      <div className="top-actions">
        <button className="icon-btn"><Search size={20} /></button>
        <button className="icon-btn"><Bell size={20} /></button>
        <div className="user-profile">
          <img src="https://ui-avatars.com/api/?name=Reviewer+One&background=E0E8F8&color=1A1D21" alt="avatar" className="avatar" />
          <span className="user-name">Reviewer</span>
        </div>
      </div>
    </div>
  );
};

const Layout = ({ children }) => {
  return (
    <div className="app-container">
      <Sidebar />
      <div className="main-content">
        <TopBar />
        <div className="page-container">
          {children}
        </div>
      </div>
    </div>
  );
};

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/dataset" element={<div className="promo-card"><h2 className="promo-title">Dataset View</h2><p className="promo-text">Mock view for dataset.</p></div>} />
          <Route path="/generation" element={<div className="promo-card"><h2 className="promo-title">Generation</h2><p className="promo-text">Trigger LLM pipeline.</p></div>} />
          <Route path="/verification" element={<Verification />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
