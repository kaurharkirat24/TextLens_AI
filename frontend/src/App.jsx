import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout/Layout';
import UploadPage from './pages/UploadPage/UploadPage';
import DashboardPage from './pages/DashboardPage/DashboardPage';
import QAPage from './pages/QAPage/QAPage';
import ReportsPage from './pages/ReportsPage/ReportsPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/qa" element={<QAPage />} />
          <Route path="/reports" element={<ReportsPage />} />
          <Route path="/" element={<Navigate to="/upload" replace />} />
          <Route path="*" element={<Navigate to="/upload" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
