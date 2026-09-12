import React, { useState } from 'react';
import { Navigate } from 'react-router-dom';
import LoginForm from './LoginForm';
import RegisterForm from './RegisterForm';
import { useAuth } from '../context/AuthContext';

export default function AuthPage() {
  const { token, login } = useAuth();
  const [isLogin, setIsLogin] = useState(true);

  // Debug log to check token value
  console.log('AuthPage: token=', token);

  // Only redirect if token is explicitly non-null and non-empty
  if (token && token !== 'null' && token !== '') {
    console.log('AuthPage: Redirecting to /app');
    return <Navigate to="/app" replace />;
  }

  const handleAuth = (t) => {
    console.log('AuthPage: handleAuth called with token=', t);
    login(t);
  };

  return (
    <div className="image-background w-full">
      <div className="h-[100dvh] flex items-center justify-center text-white">
        <div className="w-full max-w-sm rounded-2xl p-4 space-y-4 backdrop-blur-xl shadow-gray-400/30 shadow-md bg-gray-900/30 transition-height duration-300 ease-in-out">
          <div className="text-center">
            <h1 className="text-2xl font-bold">Друг, который <span className="italic">просто</span> слушает.</h1>
            <p className="text-sm text-gray-300">100% анонимно. Без советов и осуждения.</p>
          </div>

          <div className="space-y-3">
            <button className="btn-universal gr-yandex w-full py-3 text-lg font-semibold flex items-center justify-center">
              <img src="/src/assets/icons/yandex.svg" alt="Yandex" className="mr-2 h-6 w-6" />
              Войти через Yandex
            </button>
            <button className="btn-universal gr-vk w-full py-3 text-lg font-semibold flex items-center justify-center">
              <img src="/src/assets/icons/vk.svg" alt="VK" className="mr-2 h-6 w-6" />
              Войти через VK
            </button>
            <button className="btn-universal gr-mailru w-full py-3 text-lg font-semibold flex items-center justify-center">
              <img src="/src/assets/icons/mailru.svg" alt="Mail.ru" className="mr-2 h-6 w-6" />
              Войти через Mail Ru
            </button>
          </div>

          <div className="relative flex items-center">
            <div className="flex-grow border-t border-gray-600"></div>
            <span className="flex-shrink mx-4 text-gray-400 text-sm">или</span>
            <div className="flex-grow border-t border-gray-600"></div>
          </div>

          {/* Тоггл-переключатель */}
          <div className="relative flex rounded-md">
            <button
              onClick={() => {
                console.log('AuthPage: Switching to Login');
                setIsLogin(true);
              }}
              className={`flex-1 py-2 text-center font-semibold transition-colors ${
                isLogin ? 'text-white' : 'text-subtext0'
              }`}
            >
              Вход
            </button>
            <button
              onClick={() => {
                console.log('AuthPage: Switching to Register');
                setIsLogin(false);
              }}
              className={`flex-1 py-2 text-center font-semibold transition-colors ${
                !isLogin ? 'text-white' : 'text-subtext0'
              }`}
            >
              Регистрация
            </button>
            <span
              className={`absolute bottom-0 h-0.5 bg-white transition-all duration-300 ease-in-out ${
                isLogin ? 'left-0 w-1/2' : 'left-1/2 w-1/2'
              }`}
            />
          </div>

          {/* Формы */}
          {isLogin ? (
            <LoginForm onLogin={handleAuth} />
          ) : (
            <RegisterForm onLogin={handleAuth} />
          )}
        </div>
      </div>
    </div>
  );
}
