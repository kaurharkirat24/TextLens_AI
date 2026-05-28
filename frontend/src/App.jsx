import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import ErrorBoundary from './components/ErrorBoundary';
import Layout from './components/Layout/Layout';
import UploadPage from './pages/UploadPage/UploadPage';
import DashboardPage from './pages/DashboardPage/DashboardPage';
import QAPage from './pages/QAPage/QAPage';
import ReportsPage from './pages/ReportsPage/ReportsPage';
import LandingPage from './pages/LandingPage/LandingPage';

export default function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route element={<Layout />}>
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/qa" element={<QAPage />} />
            <Route path="/reports" element={<ReportsPage />} />
            <Route path="*" element={<Navigate to="/upload" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  );
}
