import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Universe from './components/Universe';
import AuthPage from './components/AuthPage';
import LandingPage from './components/LandingPage';
import { AuthProvider } from './context/AuthContext';

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/"     element={<LandingPage />} />
          <Route path="/app"  element={<Universe />} />
          <Route path="/auth" element={<AuthPage />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
