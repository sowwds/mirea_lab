import axios from 'axios';

// Safely access import.meta.env.VITE_API_URL
const API = import.meta.env.VITE_API_URL || 'http://localhost:8080';

if (!import.meta.env.VITE_API_URL) {
  console.warn('api: VITE_API_URL is not defined in .env, using fallback:', API);
}

const api = axios.create({
  baseURL: API,
  withCredentials: true,
});

export const setAuthHeader = (token) => {
  if (token) api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
  else delete api.defaults.headers.common['Authorization'];
};

let accessToken = localStorage.getItem('access_token');
setAuthHeader(accessToken);

// Interceptor for handling 401 errors and refreshing token
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    if (error.response?.status === 401 && !originalRequest._retry) {
      console.log('api: 401 detected, attempting to refresh token');
      originalRequest._retry = true; // Mark as retried to avoid infinite loop
      try {
        // const { data } = await api.post('/auth/refresh');
        console.log('api: Token refresPed successfully', { access_token: data.access_token.substring(0, 10) + '...' });
        const newAccessToken = data.access_token;
        localStorage.setItem('access_token', newAccessToken);
        setAuthHeader(newAccessToken);
        // Retry the original request with new token
        originalRequest.headers['Authorization'] = `Bearer ${newAccessToken}`;
        return api(originalRequest);
      } catch (refreshError) {
        console.error('api: Refresh token failed', refreshError.message, refreshError.response?.status);
        localStorage.removeItem('access_token');
        setAuthHeader(null);
        throw refreshError;
      }
    }
    console.error('api: Error in request', error.message, error.response?.status);
    throw error;
  }
);

export const login = async (payload) => {
  console.log('api: Sending login request to /auth/login');
  const { data } = await api.post('/auth/login', payload);
  console.log('api: Login successful, access_token:', data.access_token.substring(0, 10) + '...');
  return data.access_token;
};

export const register = async (payload) => {
  console.log('api: Sending register request to /auth/register');
  const { data } = await api.post('/auth/register', payload);
  console.log('api: Register successful, access_token:', data.access_token.substring(0, 10) + '...');
  return data.access_token;
};

export const sendMessage = async (prompt) => {
  console.log('sendMessage: Sending POST /chat/message with prompt', prompt);
  try {
    const response = await api.post('/chat/message', { prompt });
    console.log('sendMessage: Response received', response.status, response.data);
    return response;
  } catch (error) {
    console.error('sendMessage: Error sending message', error.message, error.response?.status);
    throw error;
  }
};

export const getChatRecent = async (limit = 20) => {
  const token = localStorage.getItem('access_token');
  console.log('getChatRecent: Fetching recent history with token', token?.substring(0, 10) + '...', 'limit:', limit);
  try {
    const res = await api.get(`/chat/recent?limit=${limit}`);
    if (res.status !== 200) throw new Error(`Request failed with status code ${res.status}`);
    const data = res.data;
    const history = data.messages.map((m) => ({
      role: m.role === 'user' ? 'user' : 'bot',
      text: m.content,
      event_id: m.event_id,
    }));
    console.log('getChatRecent: Recent history fetched successfully', history);
    return { messages: history, is_full: data.messages.length === limit };
  } catch (error) {
    console.error('getChatRecent: Error fetching recent messages', error.message, error.response?.status);
    throw error;
  }
};

export const getChatHistory = async (limit = 20, before_id) => {
  const token = localStorage.getItem('access_token');
  console.log('getChatHistory: Fetching history page with token', token?.substring(0, 10) + '...', 'limit:', limit, 'before_id:', before_id);
  try {
    const url = before_id ? `/chat/history?limit=${limit}&before_id=${before_id}` : `/chat/history?limit=${limit}`;
    const res = await api.get(url);
    if (res.status !== 200) throw new Error(`Request failed with status code ${res.status}`);
    const data = res.data;
    const history = data.messages.map((m) => ({
      role: m.role === 'user' ? 'user' : 'bot',
      text: m.content,
      event_id: m.event_id,
    }));
    console.log('getChatHistory: History page fetched successfully', data);
    const next_before_id = data.messages.length > 0 ? data.messages[0].event_id : null;
    return {
      messages: history,
      next_before_id,
      has_more: data.has_more,
    };
  } catch (error) {
    console.error('getChatHistory: Error fetching history', error.message, error.response?.status);
    throw error;
  }
};

export const listenChatStream = (onMessage, onError) => {
  const controller = new AbortController();
  const decoder = new TextDecoder();

  const connect = () => {
    const token = localStorage.getItem('access_token');
    if (!token) {
      console.log('listenChatStream: No token, triggering onError');
      onError?.('No access token available');
      setTimeout(connect, 3000);
      return;
    }
    console.log('listenChatStream: Attempting connection with token', token.substring(0, 10) + '...');
    fetch(`${API}/chat/stream`, {
      method: 'GET',
      headers: { Authorization: `Bearer ${token}` },
      credentials: 'include',
      signal: controller.signal,
    })
      .then((res) => {
        if (!res.ok) {
          console.error('listenChatStream: Connection failed with status', res.status);
          throw new Error(res.statusText);
        }
        console.log('listenChatStream: Connection established');
        const reader = res.body.getReader();
        const pump = () =>
          reader.read().then(({ done, value }) => {
            if (done) {
              console.log('listenChatStream: Stream closed, reconnecting in 3s');
              setTimeout(connect, 3000);
              return;
            }
            let buf = '';
            buf += decoder.decode(value, { stream: true });
            console.log('listenChatStream: Received data chunk', buf);
            const parts = buf.split('\n\n');
            buf = parts.pop() || '';
            parts.forEach((part) => {
              const lines = part.split('\n');
              let message = '';
              for (const line of lines) {
                if (line.startsWith('data:')) {
                  message = line.slice(5).trim();
                  break;
                }
              }
              if (message !== undefined) {
                console.log('listenChatStream: Processed message', message);
                onMessage(message, false);
              }
            });
            return pump();
          })
          .catch((err) => {
            if (err.name === 'AbortError') {
              console.log('listenChatStream: Connection aborted');
              return;
            }
            console.error('listenChatStream: Error in pump', err.message);
            onError?.(err.message);
            setTimeout(connect, 3000);
          });
        pump();
      })
      .catch((err) => {
        if (err.name === 'AbortError') {
          console.log('listenChatStream: Connection aborted');
          return;
        }
        console.error('listenChatStream: Connection error', err.message);
        onError?.(err.message);
        setTimeout(connect, 3000);
      });
  };

  connect();

  return () => {
    console.log('listenChatStream: Aborting connection');
    controller.abort();
  };
};

export const logout = async () => {
  try {
    console.log('api: Sending logout request to /auth/logout');
    const response = await api.post('/auth/logout');
    console.log('api: Logout successful, status:', response.status);
    localStorage.removeItem('access_token');
    setAuthHeader(null);
  } catch (error) {
    console.error('api: Logout error', error);
    localStorage.removeItem('access_token');
    setAuthHeader(null);
    throw error;
  }
};

export const getProfile = async () => {
  try {
    console.log('api: Fetching profile from /auth/profile');
    const { data } = await api.get('/auth/profile');
    console.log('api: Profile fetched successfully', data);
    return {
      fullName: data.fullName || data.username || 'Nickname',
      email: data.email || 'Email address',
    };
  } catch (error) {
    console.error('api: Error fetching profile', error);
    return {
      fullName: 'Nickname',
      email: 'Email address',
    };
  }
};
