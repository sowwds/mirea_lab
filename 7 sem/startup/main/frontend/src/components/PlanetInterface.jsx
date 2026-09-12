import React, { useState, useEffect, useRef } from 'react';
import { ArrowsPointingOutIcon, ArrowsPointingInIcon, ArrowUpIcon, ArrowDownIcon, ArrowRightIcon } from '@heroicons/react/24/solid';
import { planetConfigs } from './planetLoader';
import { getEmotionData, getTopicsByEmotion, getAIAdvice, getGradient, getEmotionColor } from '../services/emotionData';

// Компонент для анимированного прогресс-бара
const AnimatedProgressBar = ({ value, gradient, timeRange, isExpanded }) => {
  const [animatedValue, setAnimatedValue] = useState(0);

  useEffect(() => {
    console.log('ProgressBar - value:', value, 'gradient:', gradient, 'timeRange:', timeRange);
    setAnimatedValue(0);
    const timer = setTimeout(() => {
      setAnimatedValue(value);
    }, 50);

    return () => clearTimeout(timer);
  }, [value, timeRange]);

  return (
    <div className={`absolute ${isExpanded ? 'bg-black/70' : 'bg-transparent'} inset-0 rounded-t-3xl h-full overflow-hidden z-0 transition-colors duration-300`}>
      <div
        className="h-full transition-all duration-700 ease-out"
        style={{
          width: `${animatedValue}%`,
          background: gradient
        }}
      />
    </div>
  );
};

const PlanetInterface = ({ planetIndex, planetData, onClose }) => {
  const [timeRange, setTimeRange] = useState('week');
  const [isVisible, setIsVisible] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [bubbleStyle, setBubbleStyle] = useState({ left: 0, width: 0 });
  const desktopToggleContainerRef = useRef(null);
  const mobileToggleContainerRef = useRef(null);
  const attemptCountRef = useRef(0);

  const currentPlanet = planetConfigs[planetIndex + 1]; // +1 to skip central planet
  const emotionKey = currentPlanet.emotion.toLowerCase();

  const emotionData = getEmotionData(emotionKey);
  const topics = getTopicsByEmotion(emotionKey);
  const aiAdvice = getAIAdvice(emotionKey);

  // Animation for sliding in and expansion
  useEffect(() => {
    setIsVisible(true);
    return () => setIsVisible(false); // Reset on unmount
  }, []);

  // Функция для обновления позиции и ширины пузырька
  const updateBubblePosition = (containerRef, isDesktop = false) => {
    if (containerRef.current && attemptCountRef.current < 10) {
      const button = containerRef.current.querySelector(`button[data-range="${timeRange}"]`);
      if (button && containerRef.current.getBoundingClientRect().width > 0) {
        const buttonRect = button.getBoundingClientRect();
        const containerRect = containerRef.current.getBoundingClientRect();
        const left = buttonRect.left - containerRect.left;
        const width = buttonRect.width;

        console.log(`Attempt ${attemptCountRef.current} - Bubble for ${timeRange} (${isDesktop ? 'desktop' : 'mobile'}): left=${left}, width=${width}`);

        if (width > 0) {
          setBubbleStyle({ left, width });
          attemptCountRef.current = 0; // Сбрасываем счетчик при успешном обновлении
        } else {
          attemptCountRef.current += 1;
        }
      } else {
        attemptCountRef.current += 1;
      }
    }
  };

  // Обновление пузырька после анимации панели
  useEffect(() => {
    if (isVisible) {
      attemptCountRef.current = 0; // Сбрасываем счетчик попыток
      const interval = setInterval(() => {
        const containerRef = window.innerWidth >= 640 ? desktopToggleContainerRef : mobileToggleContainerRef;
        updateBubblePosition(containerRef, window.innerWidth >= 640);
        if (bubbleStyle.width > 0 || attemptCountRef.current >= 10) {
          clearInterval(interval); // Останавливаем, если успех или превышен лимит
        }
      }, 100);

      const timer = setTimeout(() => {
        clearInterval(interval); // Гарантируем остановку после 1500ms
      }, 1500); // Увеличенная задержка для десктопной анимации

      return () => {
        clearInterval(interval);
        clearTimeout(timer);
      };
    }
  }, [isVisible]);

  // Обновление пузырька при изменении timeRange
  useEffect(() => {
    if (desktopToggleContainerRef.current || mobileToggleContainerRef.current) {
      attemptCountRef.current = 0; // Сбрасываем счетчик попыток
      const interval = setInterval(() => {
        const containerRef = window.innerWidth >= 640 ? desktopToggleContainerRef : mobileToggleContainerRef;
        updateBubblePosition(containerRef, window.innerWidth >= 640);
        if (bubbleStyle.width > 0 || attemptCountRef.current >= 10) {
          clearInterval(interval);
        }
      }, 0);

      return () => clearInterval(interval);
    }
  }, [timeRange]);

  // Отслеживание изменений размеров окна
  useEffect(() => {
    const handleResize = () => {
      attemptCountRef.current = 0;
      const containerRef = window.innerWidth >= 640 ? desktopToggleContainerRef : mobileToggleContainerRef;
      updateBubblePosition(containerRef, window.innerWidth >= 640);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const toggleExpand = () => {
    setIsExpanded((prev) => !prev);
  };

  return (
    <>
      {/* Desktop version: Left, vertically centered */}
      <div
        className="hidden md:flex fixed top-[64px] bottom-[64px] left-[10px] z-50 w-[min(400px,90vw)] transition-transform duration-500 ease-[power2.inOut] will-change-transform"
        style={{
          transform: isVisible ? 'translateX(0)' : 'translateX(-100%)',
        }}
      >
        <div className="flex flex-col h-full rounded-4xl overflow-hidden border-r border-t border-b border-white/10">
          {/* Header with planet name, trend, percentage, and time range toggle */}
          <div className="relative flex flex-col gap-1 min-h-auto" style={{ backgroundColor: getEmotionColor(emotionKey, 0.2) }}>
            {/* Progress bar as background */}
            <AnimatedProgressBar
              value={emotionData[timeRange].value}
              gradient={getGradient(emotionKey)}
              timeRange={timeRange}
              isExpanded={false}
            />
            {/* Content overlay */}
            <div className="relative z-10 backdrop-blur-sm transition-colors duration-500">
              <div className="m-4">
                <div className="flex items-center justify-center">
                  <div className="flex items-center gap-2">
                    <h1 className="text-xl font-bold text-white">{currentPlanet.name}</h1>
                    <div className="flex items-center gap-1 bg-black/20 p-1 rounded-md">
                      {emotionData[timeRange].trend === 'up' ? (
                        <ArrowUpIcon className="w-3 h-3 text-green-400" />
                      ) : emotionData[timeRange].trend === 'down' ? (
                        <ArrowDownIcon className="w-3 h-3 text-red-400" />
                      ) : (
                        <ArrowRightIcon className="w-3 h-3 text-blue-400" />
                      )}
                      <span className="font-medium text-white text-[10px]">{emotionData[timeRange].value}%</span>
                    </div>
                  </div>
                </div>
                {/* Time range toggle */}
                <div className="relative flex gap-1 justify-center mt-1" ref={desktopToggleContainerRef}>
                  <div
                    className="absolute h-6 rounded-md bg-gradient-to-r from-white/20 to-white/40 transition-all duration-300 ease-in-out"
                    style={{
                      left: bubbleStyle.left,
                      width: bubbleStyle.width,
                      top: 0,
                      boxShadow: '0 0 8px rgba(255, 255, 255, 0.3)',
                    }}
                  />
                  {['day', 'week', 'month', 'all'].map((range) => (
                    <button
                      key={range}
                      data-range={range}
                      onClick={() => setTimeRange(range)}
                      className={`relative px-2 py-1 rounded-md text-[10px] min-w-[40px] transition-colors duration-300 ${
                        timeRange === range ? 'text-white font-medium' : 'text-white/60 hover:text-white'
                      }`}
                    >
                      {range === 'day' ? 'День' : range === 'week' ? 'Неделя' : range === 'month' ? 'Месяц' : 'Все время'}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Scrollable content */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar bg-black/70 backdrop-blur-xl">
            {/* Advice */}
            <div className="bg-white/5 rounded-lg p-3">
              <h2 className="text-base font-semibold text-white mb-2">Совет от Serenity</h2>
              <div className="bg-white/5 rounded-md p-2">
                <p className="text-white/80 text-xs leading-relaxed">{aiAdvice}</p>
              </div>
            </div>

            {/* Topics */}
            <div className="bg-white/5 rounded-lg p-3">
              <h2 className="text-base font-semibold text-white mb-2">Последние топики</h2>
              <div className="space-y-3 scrollbar">
                {topics.map((topic, index) => (
                  <div
                    key={index}
                    className="bg-white/5 rounded-md p-3 border-l-4"
                    style={{ borderColor: getEmotionColor(emotionKey) }}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <span className="text-white/60 text-xs">{topic.date}</span>
                      <span
                        className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                          topic.impact.startsWith('+') ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                        }`}
                      >
                        {topic.impact}
                      </span>
                    </div>
                    <p className="text-white/80 text-xs">{topic.summary}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile version: Slides up from bottom */}
      <div
        className="sm:hidden fixed left-0 right-0 z-50 border-t border-white/10 rounded-t-3xl overflow-hidden transition-transform duration-[500ms] ease-[power2.inOut]"
        style={{
          transform: isVisible
            ? isExpanded
              ? 'translateY(calc(0% + 72px))'
              : 'translateY(calc(100% - 90px))'
            : 'translateY(100%)',
          top: isExpanded ? '0px' : 'calc(100% - 90px)',
          height: isExpanded ? 'calc(100% - 72px)' : '90px',
          paddingBottom: 'env(safe-area-inset-bottom)',
        }}
      >
        <div className="flex flex-col h-full">
          {/* Header with planet name, trend, percentage, button, and time range toggle */}
          <div className="relative flex flex-col gap-1 min-h-auto" style={{ backgroundColor: getEmotionColor(emotionKey, 0.2) }}>
            {/* Progress bar as background */}
            <AnimatedProgressBar
              value={emotionData[timeRange].value}
              gradient={getGradient(emotionKey)}
              timeRange={timeRange}
              isExpanded={isExpanded}
            />
            {/* Content overlay */}
            <div className="relative z-10 backdrop-blur-sm bg-black/20 transition-colors duration-500">
              <div className="m-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <h1 className="text-xl font-bold text-white">{currentPlanet.name}</h1>
                    <div className="flex items-center gap-1 bg-black/20 p-1 rounded-md">
                      {emotionData[timeRange].trend === 'up' ? (
                        <ArrowUpIcon className="w-3 h-3 text-green-400" />
                      ) : emotionData[timeRange].trend === 'down' ? (
                        <ArrowDownIcon className="w-3 h-3 text-red-400" />
                      ) : (
                        <ArrowRightIcon className="w-3 h-3 text-blue-400" />
                      )}
                      <span className="font-medium text-white text-[10px]">{emotionData[timeRange].value}%</span>
                    </div>
                  </div>
                  <button className="btn-universal gr2 !rounded-full p-2" onClick={toggleExpand}>
                    {isExpanded ? (
                      <ArrowsPointingInIcon className="w-5 h-5 text-white" />
                    ) : (
                      <ArrowsPointingOutIcon className="w-5 h-5 text-white" />
                    )}
                  </button>
                </div>
                {/* Time range toggle */}
                <div className="relative flex gap-1 justify-center mt-1" ref={mobileToggleContainerRef}>
                  <div
                    className="absolute h-6 rounded-md bg-gradient-to-r from-white/20 to-white/40 transition-all duration-300 ease-in-out"
                    style={{
                      left: bubbleStyle.left,
                      width: bubbleStyle.width,
                      top: 0,
                      boxShadow: '0 0 8px rgba(255, 255, 255, 0.3)',
                    }}
                  />
                  {['day', 'week', 'month', 'all'].map((range) => (
                    <button
                      key={range}
                      data-range={range}
                      onClick={() => setTimeRange(range)}
                      className={`relative px-2 py-1 rounded-md text-[10px] min-w-[40px] transition-colors duration-300 ${
                        timeRange === range ? 'text-white font-medium' : 'text-white/60 hover:text-white'
                      }`}
                    >
                      {range === 'day' ? 'День' : range === 'week' ? 'Неделя' : range === 'month' ? 'Месяц' : 'Все время'}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Scrollable content */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar bg-black/70 backdrop-blur-xl">
            {/* Advice */}
            <div className="bg-white/5 rounded-lg p-3">
              <h2 className="text-base font-semibold text-white mb-2">Совет от Serenity</h2>
              <div className="bg-white/5 rounded-md p-2">
                <p className="text-white/80 text-xs leading-relaxed">{aiAdvice}</p>
              </div>
            </div>

            {/* Topics */}
            <div className="bg-white/5 rounded-lg p-3">
              <h2 className="text-base font-semibold text-white mb-2">Последние топики</h2>
              <div className="space-y-3 scrollbar">
                {topics.map((topic, index) => (
                  <div
                    key={index}
                    className="bg-white/5 rounded-md p-3 border-l-4"
                    style={{ borderColor: getEmotionColor(emotionKey) }}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <span className="text-white/60 text-xs">{topic.date}</span>
                      <span
                        className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                          topic.impact.startsWith('+') ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                        }`}
                      >
                        {topic.impact}
                      </span>
                    </div>
                    <p className="text-white/80 text-xs">{topic.summary}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default PlanetInterface;
