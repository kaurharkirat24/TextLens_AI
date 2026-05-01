import { MessageSquareText, Lock } from 'lucide-react';
import './PlaceholderPage.css';

export default function QAPage() {
  return (
    <div className="placeholder-page animate-fadeIn">
      <div className="placeholder-card card">
        <div className="placeholder-icon placeholder-icon--purple">
          <MessageSquareText size={32} />
        </div>
        <h2>Q&A</h2>
        <p>Ask natural language questions about your data and get grounded, cited answers.</p>
        <span className="placeholder-phase">
          <Lock size={12} /> Coming in Phase 3
        </span>
      </div>
    </div>
  );
}
