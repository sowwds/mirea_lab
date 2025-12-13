import { useState, useEffect } from 'react';
import api from '../services/api';

interface DefectFormProps {
  defect?: any; // Existing defect data for editing
  onSave: () => void; // Callback after successful save
  onCancel: () => void; // Callback to cancel form
}

const DefectForm = ({ defect, onSave, onCancel }: DefectFormProps) => {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [priority, setPriority] = useState('LOW');
  const [status, setStatus] = useState('NEW');
  const [assigneeId, setAssigneeId] = useState('');
  const [error, setError] = useState('');
  const [users, setUsers] = useState<any[]>([]); // To populate assignee dropdown

  useEffect(() => {
    // Fetch users for assignee dropdown
    const fetchUsers = async () => {
      try {
        const response = await api.get('/users'); // Assuming a /users endpoint exists for fetching all users
        setUsers(response.data);
      } catch (err) {
        console.error('Не удалось загрузить пользователей:', err);
      }
    };
    fetchUsers();

    if (defect) {
      setTitle(defect.title);
      setDescription(defect.description || '');
      setPriority(defect.priority);
      setStatus(defect.status);
      setAssigneeId(defect.assigneeId);
    }
  }, [defect]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    const defectData = {
      title,
      description,
      priority,
      status,
      assigneeId,
    };

    try {
      if (defect) {
        // Update existing defect
        await api.put(`/defects/${defect.id}`, defectData);
      } else {
        // Create new defect
        await api.post('/defects', defectData);
      }
      onSave(); // Notify parent component of successful save
    } catch (err) {
      setError('Не удалось сохранить дефект.');
      console.error(err);
    }
  };

  return (
    <div className="card mt-4">
      <div className="card-body">
        <h5 className="card-title">{defect ? 'Редактировать дефект' : 'Создать новый дефект'}</h5>
        <form onSubmit={handleSubmit}>
          {error && <div className="alert alert-danger">{error}</div>}
          <div className="mb-3">
            <label htmlFor="title" className="form-label">Название</label>
            <input
              type="text"
              className="form-control"
              id="title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              required
            />
          </div>
          <div className="mb-3">
            <label htmlFor="description" className="form-label">Описание</label>
            <textarea
              className="form-control"
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            ></textarea>
          </div>
          <div className="mb-3">
            <label htmlFor="priority" className="form-label">Приоритет</label>
            <select
              className="form-select"
              id="priority"
              value={priority}
              onChange={(e) => setPriority(e.target.value)}
            >
              <option value="LOW">Низкий</option>
              <option value="MEDIUM">Средний</option>
              <option value="HIGH">Высокий</option>
            </select>
          </div>
          <div className="mb-3">
            <label htmlFor="status" className="form-label">Статус</label>
            <select
              className="form-select"
              id="status"
              value={status}
              onChange={(e) => setStatus(e.target.value)}
            >
              <option value="NEW">Новая</option>
              <option value="IN_PROGRESS">В работе</option>
              <option value="UNDER_REVIEW">На проверке</option>
              <option value="CLOSED">Закрыта</option>
              <option value="CANCELLED">Отменена</option>
            </select>
          </div>
          <div className="mb-3">
            <label htmlFor="assignee" className="form-label">Исполнитель</label>
            <select
              className="form-select"
              id="assignee"
              value={assigneeId}
              onChange={(e) => setAssigneeId(e.target.value)}
              required
            >
              <option value="">Выберите исполнителя</option>
              {users.map((user) => (
                <option key={user.id} value={user.id}>
                  {user.email}
                </option>
              ))}
            </select>
          </div>
          <button type="submit" className="btn btn-primary me-2">Сохранить</button>
          <button type="button" className="btn btn-secondary" onClick={onCancel}>Отмена</button>
        </form>
      </div>
    </div>
  );
};

export default DefectForm;
