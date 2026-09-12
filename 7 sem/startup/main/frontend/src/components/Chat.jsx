import React, { useEffect, useRef, useState } from 'react';
import { PaperAirplaneIcon, ArrowPathIcon } from '@heroicons/react/24/solid';
import { sendMessage, listenChatStream, getChatRecent, getChatHistory } from '../services/api';
import SentMessage from './SentMessage';
import ReceivedMessage from './ReceivedMessage';

const Chat = ({ onBack }) => {
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState('');
  const [currentBotMessage, setCurrentBotMessage] = useState('');
  const [isWaitingForResponse, setIsWaitingForResponse] = useState(false);
  const [isRecentFull, setIsRecentFull] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [nextBeforeId, setNextBeforeId] = useState(null);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [isMounted, setIsMounted] = useState(false);
  const messagesRef = useRef(null);
  const textareaRef = useRef(null);
  const sendButtonRef = useRef(null); // Добавляем ref для кнопки
  const isHistoryLoaded = useRef(false);
  const isSending = useRef(false);
  const oldScrollHeight = useRef(0);

  const scrollToBottom = () => {
    requestAnimationFrame(() => {
      const el = messagesRef.current;
      if (!el) return;
      el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
    });
  };

  const loadHistory = async () => {
    if (isHistoryLoaded.current) {
      console.log('loadHistory: Skipping, history already loaded');
      return;
    }
    isHistoryLoaded.current = true;
    try {
      const data = await getChatRecent(50);
      console.log('loadHistory: Recent history loaded', data);
      setMessages(data.messages.reverse());
      setIsRecentFull(data.is_full);
      if (data.is_full && data.messages.length > 0) {
        setNextBeforeId(data.messages[0].event_id);
        console.log('loadHistory: Set nextBeforeId to', data.messages[0].event_id);
      }
      if (data.messages.length === 0) {
        console.log('loadHistory: Empty messages array, adding banner');
        setMessages((prev) => [
          ...prev,
          { role: 'banner', text: 'Добрый день!\nЭто начало вашего диалога с Serenity AI, учтите, что <тут текст дисклеймера>' },
        ]);
      }
    } catch (err) {
      console.error('loadHistory: Error', err.message);
      setMessages((prev) => {
        if (prev.some((msg) => msg.role === 'banner')) {
          return prev;
        }
        return [
          ...prev,
          { role: 'banner', text: 'Добрый день!\nЭто начало вашего диалога с Serenity AI, учтите, что <тут текст дисклеймера>' },
        ];
      });
    }
  };

  const loadMoreHistory = async () => {
    if (!isRecentFull || !hasMore || isLoadingHistory) {
      console.log('loadMoreHistory: Skipping', { isRecentFull, hasMore, isLoadingHistory });
      return;
    }
    setIsLoadingHistory(true);
    oldScrollHeight.current = messagesRef.current.scrollHeight;
    try {
      const data = await getChatHistory(20, nextBeforeId);
      console.log('loadMoreHistory: Additional history loaded', data);
      if (data.messages.length > 0) {
        setMessages((prev) => [...data.messages.reverse(), ...prev]);
        setNextBeforeId(data.next_before_id);
        console.log('loadMoreHistory: Set nextBeforeId to', data.next_before_id);
      }
      setHasMore(data.has_more);
    } catch (err) {
      console.error('loadMoreHistory: Error', err.message);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  const completeBotMessage = () => {
    if (currentBotMessage) {
      setMessages((m) => {
        const newMessages = [...m, { role: 'bot', text: currentBotMessage }];
        console.log('Chat: Completed bot message', newMessages);
        return newMessages;
      });
      setCurrentBotMessage('');
    }
    setIsWaitingForResponse(false);
  };

  useEffect(() => {
    console.log('Chat: Mounting, starting loadHistory and listenChatStream');
    loadHistory();
    const cleanup = listenChatStream(
      (text, isDone) => {
        console.log('Chat: Received SSE message', text, 'isDone:', isDone);
        if (text === '[DONE]') {
          console.log('Chat: Received [DONE], completing bot message');
          completeBotMessage();
          return;
        }
        if (text) {
          setCurrentBotMessage((prev) => {
            const noSpaceBefore = [',', '.', '!', '?', ':', ';', ')', ']', '"', "'"];
            const shouldAddSpace = prev && !noSpaceBefore.some((char) => text.startsWith(char));
            return prev + (shouldAddSpace ? ' ' : '') + text;
          });
        } else {
          console.log('Chat: Empty SSE message, skipping');
        }
      },
      (error) => {
        console.error('Chat: SSE error:', error);
        setIsWaitingForResponse(false);
      }
    );
    return () => {
      console.log('Chat: Unmounting, calling cleanup');
      cleanup();
    };
  }, []);

  useEffect(() => {
    console.log('Chat: Messages or currentBotMessage updated, scrolling to bottom', { messages, currentBotMessage });
    const el = messagesRef.current;
    if (!el) return;
    if (oldScrollHeight.current > 0) {
      const newScrollHeight = el.scrollHeight;
      el.scrollTop += newScrollHeight - oldScrollHeight.current;
      oldScrollHeight.current = 0;
    } else {
      scrollToBottom();
    }
  }, [messages, currentBotMessage]);

  useEffect(() => {
    const el = messagesRef.current;
    if (!el) return;
    const handleScroll = () => {
      if (el.scrollTop === 0 && isRecentFull && hasMore && !isLoadingHistory) {
        console.log('Chat: Scrolled to top, triggering loadMoreHistory');
        loadMoreHistory();
      }
    };
    el.addEventListener('scroll', handleScroll);
    return () => el.removeEventListener('scroll', handleScroll);
  }, [isRecentFull, hasMore, isLoadingHistory]);

  // Сбрасываем состояние кнопки после получения ответа
  useEffect(() => {
    if (!isWaitingForResponse && sendButtonRef.current) {
      // Удаляем класс active, если он был добавлен
      sendButtonRef.current.classList.remove('active');
      // Принудительно сбрасываем стили, связанные с hover
      sendButtonRef.current.style.transform = '';
      sendButtonRef.current.style.boxShadow = '';
      sendButtonRef.current.style.background = '';
    }
  }, [isWaitingForResponse]);

  const handleSend = async () => {
    if (!draft.trim() || isSending.current || isWaitingForResponse) return;
    isSending.current = true;
    const text = draft.trim();
    setDraft('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.focus();
    }
    completeBotMessage();
    setMessages((m) => {
      const newMessages = [...m, { role: 'user', text }];
      console.log('handleSend: Added user message', newMessages);
      return newMessages;
    });
    setIsWaitingForResponse(true);
    try {
      console.log('handleSend: Sending message', text);
      await sendMessage(text);
    } catch (e) {
      console.error('handleSend: Send error', e);
      setIsWaitingForResponse(false);
      if (e.response?.status === 401) {
        console.log('handleSend: 401 detected, logging out');
        onBack();
      }
    } finally {
      isSending.current = false;
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e) => {
    setDraft(e.target.value);
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  };

  useEffect(() => {
    setIsMounted(true);
  }, []);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  }, []);

  const handleChildClick = (e) => {
    e.stopPropagation();
  };

  return (
    <div className={`w-full bg-black/20 transition-opacity duration-500 ${isMounted ? 'opacity-100' : 'opacity-0'}`} onClick={onBack}>
      <div
        className="flex flex-col h-[100dvh] mx-auto w-full max-w-2xl"
        onClick={handleChildClick}
      >
        <div
          ref={messagesRef}
          className="chat-messages scrollbar p-4 text-lg flex-1 overflow-y-auto pt-16"
        >
          {isLoadingHistory && (
            <div className="flex items-center justify-center mb-4 text-subtext0">
              <ArrowPathIcon className="h-6 w-6 animate-spin mr-2" />
              Загрузка истории...
            </div>
          )}
          {messages.map((msg, i) => (
            msg.role === 'banner' ? (
              <div key={i} className="flex items-center justify-center h-full w-full text-subtext0 text-lg">
                <div className="text-center backdrop-blur-xl shadow-gray-400/30 shadow-md bg-gray-900/30 text-white py-3 px-4 rounded-3xl w-fit max-w-xs whitespace-pre-wrap">
                  {msg.text}
                </div>
              </div>
            ) : msg.role === 'user' ? (
              <SentMessage key={msg.event_id || i} text={msg.text} />
            ) : (
              <ReceivedMessage key={msg.event_id || i} text={msg.text} />
            )
          ))}
          {currentBotMessage && (
            <ReceivedMessage key="current-bot" text={currentBotMessage} />
          )}
        </div>

        <div className="flex items-end gap-2 bg-surface0/50 backdrop-blur-xs rounded-t-3xl sm:rounded-4xl p-2 border border-gray-600 sm:mb-5">
          <textarea
            ref={textareaRef}
            rows={1}
            className="flex-1 bg-transparent text-white placeholder-subtext0 focus:outline-none px-3 py-2 resize-none overflow-y-auto leading-tight"
            placeholder="Напишите сообщение..."
            value={draft}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            style={{ minHeight: '40px', maxHeight: '160px' }}
          />
          <button
            ref={sendButtonRef}
            className={`btn-universal gr2 p-2 !rounded-full ${isWaitingForResponse ? 'disabled' : ''}`}
            onClick={handleSend}
            disabled={isWaitingForResponse}
          >
            {isWaitingForResponse ? (
              <ArrowPathIcon className="h-7 w-7 animate-spin" />
            ) : (
              <PaperAirplaneIcon className="h-7 w-7" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chat;
