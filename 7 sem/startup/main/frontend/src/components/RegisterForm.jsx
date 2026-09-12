import React, { useState } from 'react';
import { EyeIcon, EyeSlashIcon } from '@heroicons/react/24/solid';
import { register } from '../services/api';

const EMAIL_REGEX = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;

export default function RegisterForm({ onLogin }) {
  const [nickname, setNickname] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPwd, setShowPwd] = useState(false);
  const [nickErr, setNickErr] = useState('');
  const [emailErr, setEmailErr] = useState('');
  const [pwdErr, setPwdErr] = useState('');
  const [serverErr, setServerErr] = useState('');

  console.log('RegisterForm: Rendering');

  /* ---------- валидация ---------- */
  const validateNickname = () => {
    setNickErr('');
    if (!nickname) return;
    if (nickname.length < 4 || nickname.length > 32)
      setNickErr('Никнейм 4–32 символов');
  };

  const validateEmail = () => {
    setEmailErr('');
    if (!email) return;
    if (!EMAIL_REGEX.test(email)) setEmailErr('Неверный формат email');
  };

  const validatePassword = () => {
    setPwdErr('');
    if (!password) return;
    if (password.length < 6 || password.length > 32)
      setPwdErr('Пароль 6–32 символов');
  };

  const isValid = !nickErr && !emailErr && !pwdErr && nickname && email && password;

  /* ---------- отправка ---------- */
  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log('RegisterForm: Submitting with email=', email, 'nickname=', nickname);
    setServerErr('');
    try {
      const token = await register({ nickname, email, password });
      console.log('RegisterForm: Registration successful, token=', token);
      onLogin(token);
    } catch (err) {
      console.error('RegisterForm: Error=', err);
      setServerErr(
        err.response?.data?.error || err.message || 'Ошибка регистрации'
      );
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <input
          type="text"
          placeholder="Nickname"
          value={nickname}
          onChange={(e) => setNickname(e.target.value)}
          onBlur={validateNickname}
          className="w-full p-2 bg-transparent border border-overlay0 rounded-xl placeholder-subtext0 focus:outline-none"
          required
        />
        {nickErr && <p className="text-xs text-red-400 px-1">{nickErr}</p>}
      </div>

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
      {pwdErr && <p className="text-xs text-red-400 px-1">{pwdErr}</p>}

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
        Зарегистрироваться
      </button>
    </form>
  );
}
