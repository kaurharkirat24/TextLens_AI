import { Component } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error('[TextLens UI] Render failure', error, info);
  }

  render() {
    if (!this.state.error) {
      return this.props.children;
    }

    return (
      <main className="error-boundary">
        <AlertTriangle size={22} />
        <h1>Something went wrong</h1>
        <p>{this.state.error?.message || 'The interface could not render this view.'}</p>
        <button className="btn btn-secondary" type="button" onClick={() => window.location.reload()}>
          <RefreshCw size={16} />
          Reload
        </button>
      </main>
    );
  }
}
