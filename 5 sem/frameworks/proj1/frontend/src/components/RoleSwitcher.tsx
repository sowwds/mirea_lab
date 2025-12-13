import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import api from '../services/api';
import { jwtDecode } from 'jwt-decode';

const RoleSwitcher = () => {
  const { token, login } = useAuth();
  const [currentRole, setCurrentRole] = useState<string | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    if (token) {
      try {
        const decodedToken: any = jwtDecode(token);
        setCurrentRole(decodedToken.role);
      } catch (err) {
        console.error('Не удалось декодировать токен:', err);
      }
    }
  }, [token]);

  const handleRoleChange = async (newRole: string) => {
    setError('');
    try {
      const response = await api.put('/users/role', { role: newRole });
      const newToken = response.data.token;
      if (newToken) {
        login(newToken); // Update the auth context with the new token
        alert(`Роль успешно изменена на ${newRole}`);
      }
    } catch (err) {
      setError('Не удалось сменить роль.');
      console.error(err);
    }
  };

  if (!token) return null;

  return (
    <div className="d-flex align-items-center">
      <span className="navbar-text me-2">Роль: {currentRole}</span>
      <div className="btn-group">
        <button type="button" className="btn btn-secondary dropdown-toggle" data-bs-toggle="dropdown" aria-expanded="false">
          Сменить роль
        </button>
        <ul className="dropdown-menu dropdown-menu-end">
          <li><button className="dropdown-item" onClick={() => handleRoleChange('ENGINEER')}>Инженер</button></li>
          <li><button className="dropdown-item" onClick={() => handleRoleChange('MANAGER')}>Менеджер</button></li>
          <li><button className="dropdown-item" onClick={() => handleRoleChange('OBSERVER')}>Наблюдатель</button></li>
        </ul>
      </div>
      {error && <small className="text-danger ms-2">{error}</small>}
    </div>
  );
};

export default RoleSwitcher;
