// Updated AuthContext.jsx with useEffect for setAuthHeader
import React, { createContext, useContext, useState, useEffect } from 'react';
import { setAuthHeader } from '../services/api'; // Import setAuthHeader

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [token, setToken] = useState(localStorage.getItem('access_token'));

  const login = (t) => {
    localStorage.setItem('access_token', t);
    setToken(t);
  };

  const handleLogout = () => {
    localStorage.removeItem('access_token');
    setToken(null);
  };

  useEffect(() => {
    if (token) {
      setAuthHeader(token);
    } else {
      setAuthHeader(null);
    }
  }, [token]);

  return (
    <AuthContext.Provider value={{ token, login, handleLogout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
