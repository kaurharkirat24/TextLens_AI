import { useState, useEffect } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import {
  Upload, LayoutDashboard, MessageSquareText,
  FileText, Sparkles, Activity
} from 'lucide-react';
import { checkHealth } from '../../services/api';
import './Layout.css';

const NAV_ITEMS = [
  { to: '/upload',    icon: Upload,              label: 'Upload' },
  { to: '/dashboard', icon: LayoutDashboard,     label: 'Dashboard' },
  { to: '/qa',        icon: MessageSquareText,   label: 'Q&A' },
  { to: '/reports',   icon: FileText,            label: 'Reports' },
];

export default function Layout() {
  const [isConnected, setIsConnected] = useState(true);

  useEffect(() => {
    let interval;
    const check = async () => {
      try {
        await checkHealth();
        setIsConnected(true);
      } catch {
        setIsConnected(false);
      }
    };

    check();
    interval = setInterval(check, 10000); // Check every 10s

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="layout">
      {/* ── Sidebar ─────────────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="sidebar-brand-icon">
            <Sparkles size={20} />
          </div>
          <div>
            <h1 className="sidebar-brand-title">TextLens</h1>
            <span className="sidebar-brand-tag">AI</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `sidebar-link ${isActive ? 'sidebar-link--active' : ''}`
              }
            >
              <Icon size={18} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-status">
            <Activity
              size={14}
              className={`sidebar-status-dot ${!isConnected ? 'sidebar-status-dot--disconnected' : ''}`}
            />
            <span>{isConnected ? 'API Connected' : 'API Disconnected'}</span>
          </div>
        </div>
      </aside>

      {/* ── Main content ────────────────────────────────────────── */}
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
}
