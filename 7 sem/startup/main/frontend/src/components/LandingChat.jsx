import React, { useState, useEffect, useRef } from 'react';
import SentMessage from './SentMessage';
import ReceivedMessage from './ReceivedMessage';

const DEMO_SCRIPT = [
  { delay: 900,  role: 'bot',    text: 'Привет! Я Serenity. Расскажи, что тебя беспокоит — я здесь, чтобы выслушать.' },
  { delay: 2400, role: 'user',   text: 'Последнее время чувствую себя подавленным...' },
  { delay: 3200, role: 'typing' },
  { delay: 4800, role: 'bot',    text: 'Понимаю, это непросто. Ты не одинок в этом.\n\nМожешь рассказать побольше — когда это началось?' },
  { delay: 6400, role: 'user',   text: 'Наверное, после смены работы' },
  { delay: 7200, role: 'typing' },
];

const CANNED_RESPONSE = 'Это демо-версия Serenity. Зарегистрируйтесь, чтобы начать настоящий диалог!';

const TypingDots = () => (
  <div className="flex justify-start mb-4">
    <div className="backdrop-blur-xl shadow-gray-400/30 shadow-md bg-gray-900/30 text-white py-3 px-4 rounded-2xl">
      <div className="flex gap-1 items-center h-4">
        {[0, 1, 2].map(i => (
          <span
            key={i}
            className="w-1.5 h-1.5 bg-white/50 rounded-full animate-bounce"
            style={{ animationDelay: `${i * 0.18}s` }}
          />
        ))}
      </div>
    </div>
  </div>
);

const LandingChat = () => {
  const [messages, setMessages]       = useState([]);
  const [showTyping, setShowTyping]   = useState(false);
  const [inputEnabled, setInputEnabled] = useState(false);
  const [draft, setDraft]             = useState('');
  const [waiting, setWaiting]         = useState(false);
  const messagesRef = useRef(null);

  const scrollToBottom = () => {
    requestAnimationFrame(() => {
      if (messagesRef.current) {
        messagesRef.current.scrollTo({ top: messagesRef.current.scrollHeight, behavior: 'smooth' });
      }
    });
  };

  useEffect(() => {
    const timers = [];
    let enableInputTimer;

    DEMO_SCRIPT.forEach((step, i) => {
      const isLast = i === DEMO_SCRIPT.length - 1;
      if (step.role === 'typing') {
        timers.push(setTimeout(() => {
          setShowTyping(true);
          scrollToBottom();
          if (isLast) {
            enableInputTimer = setTimeout(() => setInputEnabled(true), 1400);
          }
        }, step.delay));
      } else {
        timers.push(setTimeout(() => {
          setShowTyping(false);
          setMessages(prev => [...prev, { role: step.role, text: step.text }]);
          if (isLast) {
            enableInputTimer = setTimeout(() => setInputEnabled(true), 1400);
          }
        }, step.delay));
      }
    });

    return () => {
      timers.forEach(clearTimeout);
      clearTimeout(enableInputTimer);
    };
  }, []);

  useEffect(() => { scrollToBottom(); }, [messages, showTyping]);

  const handleSend = () => {
    if (!draft.trim() || waiting || !inputEnabled) return;
    const text = draft.trim();
    setDraft('');
    setMessages(prev => [...prev, { role: 'user', text }]);
    setWaiting(true);
    setShowTyping(true);
    setTimeout(() => {
      setShowTyping(false);
      setMessages(prev => [...prev, { role: 'bot', text: CANNED_RESPONSE }]);
      setWaiting(false);
    }, 1200);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="bg-black/20 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden flex flex-col h-[420px] w-full shadow-2xl shadow-black/40">
      {/* Header */}
      <div className="px-4 py-3 bg-white/[0.04] border-b border-white/[0.07] flex items-center gap-2 flex-shrink-0">
        <span className="text-white font-semibold text-sm">✦ Serenity чат</span>
        <div className="ml-auto flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          <span className="text-green-400 text-xs">онлайн</span>
        </div>
      </div>

      {/* Messages area */}
      <div ref={messagesRef} className="flex-1 overflow-y-auto p-4 scrollbar">
        {messages.map((msg, i) =>
          msg.role === 'user'
            ? <SentMessage key={i} text={msg.text} />
            : <ReceivedMessage key={i} text={msg.text} />
        )}
        {showTyping && <TypingDots />}
      </div>

      {/* Input */}
      <div
        className={`flex items-end gap-2 bg-surface0/30 backdrop-blur-sm p-2 border-t border-white/[0.07] transition-opacity duration-500 ${
          inputEnabled ? 'opacity-100' : 'opacity-35 pointer-events-none'
        }`}
      >
        <textarea
          rows={1}
          className="flex-1 bg-transparent text-white placeholder-subtext0 focus:outline-none px-3 py-2 resize-none text-sm leading-tight"
          placeholder={inputEnabled ? 'Напишите сообщение...' : 'Подождите...'}
          value={draft}
          onChange={e => setDraft(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={!inputEnabled || waiting}
          style={{ minHeight: '36px', maxHeight: '100px' }}
        />
        <button
          className={`btn-universal gr2 p-2 !rounded-full flex-shrink-0 ${waiting ? 'disabled' : ''}`}
          onClick={handleSend}
          disabled={!inputEnabled || waiting}
        >
          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
          </svg>
        </button>
      </div>
    </div>
  );
};

export default LandingChat;
