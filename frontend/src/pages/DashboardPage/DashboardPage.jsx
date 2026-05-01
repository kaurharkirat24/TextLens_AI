import { LayoutDashboard, Lock } from 'lucide-react';
import './PlaceholderPage.css';

export default function DashboardPage() {
  return (
    <div className="placeholder-page animate-fadeIn">
      <div className="placeholder-card card">
        <div className="placeholder-icon">
          <LayoutDashboard size={32} />
        </div>
        <h2>Dashboard</h2>
        <p>Sentiment charts, topic analysis, and interactive analytics will appear here.</p>
        <span className="placeholder-phase">
          <Lock size={12} /> Coming in Phase 2
        </span>
      </div>
    </div>
  );
}
