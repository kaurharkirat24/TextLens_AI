import { useNavigate } from 'react-router-dom';
import { ArrowRight, Sparkles } from 'lucide-react';
import './LandingPage.css';

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="landing-page">
      <nav className="landing-nav">
        <div className="landing-logo">
          <Sparkles size={24} className="logo-accent" /> TextLens<span className="logo-accent">AI</span>
        </div>
        <button className="nav-cta" onClick={() => navigate('/upload')}>Go to App</button>
      </nav>

      <section className="hero-section">
        {/* Animated Background Mesh */}
        <div className="bg-mesh">
          <div className="mesh-orb orb-1"></div>
          <div className="mesh-orb orb-2"></div>
          <div className="mesh-orb orb-3"></div>
        </div>

        <div className="hero-content">
          <div className="hero-badge-pill">
            <span className="live-dot"></span> System Online
          </div>
          <h1 className="hero-title">
            Uncover the Hidden <br/><span className="highlight-text-3d">Sentiments</span> of Data
          </h1>
          <p className="hero-subtitle">
            Instantly transform raw text into actionable insights using state-of-the-art Natural Language Processing.
          </p>
          <button className="hero-cta" onClick={() => navigate('/upload')}>
            Enter Dashboard <ArrowRight size={20} />
          </button>
        </div>
        
        <div className="hero-visual">
          <div className="perspective-container">
            <div className="glass-panel">
              <div className="glass-reflection"></div>
              <img src="https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=2070&auto=format&fit=crop" alt="AI Dashboard" className="hero-image-3d" />
            </div>
            {/* Floating 3D Elements */}
            <div className="float-element abstract-1"></div>
            <div className="float-element abstract-2"></div>
            <div className="float-element abstract-3"></div>
          </div>
        </div>
      </section>
    </div>
  );
}
