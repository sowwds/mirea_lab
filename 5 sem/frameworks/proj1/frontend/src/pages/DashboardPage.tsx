import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import api from '../services/api';
import DefectForm from '../components/DefectForm'; // Assuming DefectForm exists
import { jwtDecode } from 'jwt-decode'; // To decode JWT token for role

const DashboardPage = () => {
  const { logout, token } = useAuth(); // Get token from useAuth
  const [defects, setDefects] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [editingDefect, setEditingDefect] = useState<any | null>(null);
  const [userRole, setUserRole] = useState<string | null>(null);

  useEffect(() => {
    if (token) {
      try {
        const decodedToken: any = jwtDecode(token);
        setUserRole(decodedToken.role);
      } catch (err) {
        console.error('Failed to decode token:', err);
        setUserRole(null);
      }
    }
  }, [token]);

  const fetchDefects = async () => {
    try {
      setLoading(true);
      const response = await api.get('/defects');
      setDefects(response.data);
      setError('');
    } catch (err) {
      setError('Failed to fetch defects.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDefects();
  }, []);

  const handleCreateClick = () => {
    setEditingDefect(null); // Clear any defect being edited
    setShowForm(true);
  };

  const handleEditClick = (defect: any) => {
    setEditingDefect(defect);
    setShowForm(true);
  };

  const handleDeleteClick = async (id: string) => {
    if (window.confirm('Are you sure you want to delete this defect?')) {
      try {
        await api.delete(`/defects/${id}`);
        fetchDefects(); // Refresh list
      } catch (err) {
        setError('Failed to delete defect. Check your permissions.');
        console.error(err);
      }
    }
  };

  const handleFormSave = () => {
    setShowForm(false);
    fetchDefects(); // Refresh list after save
  };

  const handleFormCancel = () => {
    setShowForm(false);
    setEditingDefect(null);
  };

  return (
    <div className="container mt-5">
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h1>Defects Dashboard</h1>
        <div>
          {userRole !== 'OBSERVER' && (
            <button className="btn btn-success me-2" onClick={handleCreateClick}>Create Defect</button>
          )}
          <button className="btn btn-danger" onClick={logout}>Logout</button>
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      {showForm && (
        <DefectForm
          defect={editingDefect}
          onSave={handleFormSave}
          onCancel={handleFormCancel}
        />
      )}

      {!showForm && (
        <>
          {loading ? (
            <p>Loading defects...</p>
          ) : defects.length === 0 ? (
            <p>No defects found. Create one!</p>
          ) : (
            <div className="table-responsive">
              <table className="table table-striped table-hover">
                <thead>
                  <tr>
                    <th>Title</th>
                    <th>Description</th>
                    <th>Priority</th>
                    <th>Status</th>
                    <th>Assignee</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {defects.map((defect) => (
                    <tr key={defect.id}>
                      <td>{defect.title}</td>
                      <td>{defect.description}</td>
                      <td>{defect.priority}</td>
                      <td>{defect.status}</td>
                      <td>{defect.assigneeEmail || defect.assigneeId}</td> {/* Display email if available */}
                      <td>
                        {userRole !== 'OBSERVER' && (
                          <>
                            <button className="btn btn-warning btn-sm me-2" onClick={() => handleEditClick(defect)}>Edit</button>
                            {userRole === 'MANAGER' && ( // Only manager can delete
                              <button className="btn btn-danger btn-sm" onClick={() => handleDeleteClick(defect.id)}>Delete</button>
                            )}
                          </>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default DashboardPage;
