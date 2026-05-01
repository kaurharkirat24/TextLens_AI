import { FileText, Lock } from 'lucide-react';
import './PlaceholderPage.css';

export default function ReportsPage() {
  return (
    <div className="placeholder-page animate-fadeIn">
      <div className="placeholder-card card">
        <div className="placeholder-icon placeholder-icon--green">
          <FileText size={32} />
        </div>
        <h2>Reports</h2>
        <p>Generate and export PDF reports, enriched CSVs, and executive summaries.</p>
        <span className="placeholder-phase">
          <Lock size={12} /> Coming in Phase 5
        </span>
      </div>
    </div>
  );
}
