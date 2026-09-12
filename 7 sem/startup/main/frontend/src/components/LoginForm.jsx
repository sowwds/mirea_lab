import React, { useState } from 'react';
import { EyeIcon, EyeSlashIcon } from '@heroicons/react/24/solid';
import { login } from '../services/api';

const EMAIL_REGEX = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;

export default function LoginForm({ onLogin }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPwd, setShowPwd] = useState(false);
  const [emailErr, setEmailErr] = useState('');
  const [passwordErr, setPasswordErr] = useState('');
  const [serverErr, setServerErr] = useState('');

  console.log('LoginForm: Rendering');

  /* ---------- валидация ---------- */
  const validateEmail = () => {
    setEmailErr('');
    if (!email) return;
    if (!EMAIL_REGEX.test(email)) setEmailErr('Неверный формат email');
  };

  const validatePassword = () => {
    setPasswordErr('');
    if (!password) return;
    if (password.length < 6 || password.length > 32)
      setPasswordErr('Пароль 6–32 символов');
  };

  const isValid = !emailErr && !passwordErr && email && password;

  /* ---------- отправка ---------- */
  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log('LoginForm: Submitting with email=', email);
    setServerErr('');
    try {
      const token = await login({ email, password });
      console.log('LoginForm: Login successful, token=', token);
      onLogin(token);
    } catch (err) {
      console.error('LoginForm: Error=', err);
      setServerErr(
        err.response?.data?.error || err.message || 'Ошибка входа'
      );
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          onBlur={validateEmail}
          className="w-full p-2 bg-transparent border border-overlay0 rounded-xl placeholder-subtext0 focus:outline-none"
          required
        />
        {emailErr && <p className="text-xs text-red-400 px-1">{emailErr}</p>}
      </div>

      <div className="relative">
        <input
          type={showPwd ? 'text' : 'password'}
          placeholder="Пароль"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onBlur={validatePassword}
          className="w-full p-2 pr-10 bg-transparent border border-overlay0 rounded-xl placeholder-subtext0 focus:outline-none"
          required
        />
        <button
          type="button"
          onClick={() => setShowPwd(!showPwd)}
          className="absolute inset-y-0 right-0 px-2 flex items-center text-gray-300 hover:text-white"
        >
          {showPwd ? <EyeSlashIcon className="w-5 h-5" /> : <EyeIcon className="w-5 h-5" />}
        </button>
      </div>
      {passwordErr && <p className="text-xs text-red-400 px-1">{passwordErr}</p>}

      {serverErr && (
        <p className="p-2 bg-red-400 rounded-md text-white text-sm text-center">
          {serverErr}
        </p>
      )}
      <button
        className={`
          btn-universal gr1
          w-full p-2
          ${isValid ? '' : 'disabled'}
        `}
        disabled={!isValid}
        type="submit"
      >
        Войти
      </button>
    </form>
  );
}
